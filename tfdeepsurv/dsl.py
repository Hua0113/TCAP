import os
import pandas as pd
import numpy as np
import tensorflow as tf

from .utils import _check_config
from .utils import _check_surv_data
from .utils import _prepare_surv_data
from .utils import concordance_index
from .utils import baseline_survival_function
from .vision import plot_train_curve, plot_surv_curve



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'




class auto_cox(object):
    """dsnn model"""
    def __init__(self,hidden_layers_nodes,en_concig={},config={},):
        super(auto_cox, self).__init__()
        ###cox_part
        self.en_config = en_concig
        tf.set_random_seed(config["seed"])
        n_input = en_concig['n_input']
        n_hidden_1 =en_concig['n_hidden_1']
        n_hidden_2 = en_concig['n_hidden_2']
        self.X = tf.placeholder(tf.float32, [None, n_input], name='X-Input')
        self.Y = tf.placeholder(tf.float32, [None, 1], name='Y-Input')


        # neural nodes
        self.input_nodes = n_hidden_2
        self.hidden_layers_nodes = hidden_layers_nodes
        assert hidden_layers_nodes[-1] == 1
        # network hyper-parameters
        _check_config(config)
        self.config = config
        # graph level random seed

        # some gobal settings
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.dropout = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)
        self.loss_alpha = tf.placeholder(tf.float32)


    ###cox part

    def _create_network(self):
        """
        Define the neural network that only includes FC layers.
        """
        scale = self.en_config["L2_reg"]
        ###auto encoder
        fc_1 = tf.layers.dense(inputs=self.X, units=self.en_config["n_hidden_1"],
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        fc_1_out = tf.nn.relu(fc_1)
        fc_1_dropout = tf.layers.dropout(inputs=fc_1_out, rate=self.dropout, training=self.training)
        fc_2 = tf.layers.dense(inputs=fc_1_dropout, units=self.en_config["n_hidden_2"],
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        fc_2_out = tf.nn.relu(fc_2)
        encoder_op = tf.layers.dropout(inputs=fc_2_out, rate=self.dropout, training=self.training)

        fc_3 = tf.layers.dense(inputs=encoder_op, units=self.en_config["n_hidden_1"],
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        fc_3_out = tf.nn.relu(fc_3)
        fc_3_dropout = tf.layers.dropout(inputs=fc_3_out, rate=self.dropout, training=self.training)
        decoder_op = tf.layers.dense(inputs=fc_3_dropout, units=self.en_config["n_input"])
        self.encoder_op = encoder_op
        self.decoder_op = decoder_op

        ###cox fc
        with tf.name_scope("hidden_layers"):
            en_x = self.encoder_op
            for i, num_nodes in enumerate(self.hidden_layers_nodes):
                en_x = tf.layers.dense(inputs=en_x, units=num_nodes, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
                if num_nodes!=1:
                    en_x = tf.nn.relu(en_x)
            self.Y_hat = en_x

    def _create_loss(self):
        """
        Define the loss function.

        Notes
        -----
        The loss function definded here is negative log of Breslow Approximation partial 
        likelihood function. See more in "Breslow N., 'Covariance analysis of censored 
        survival data, ' Biometrics 30.1(1974):89-99.".
        """
        with tf.name_scope("loss"):
            # auto_encoder_part
            en_pred = self.decoder_op
            en_true = self.X
            self.en_cost = tf.reduce_mean(tf.pow(en_true - en_pred, 2))  #
            l2_loss = tf.losses.get_regularization_loss()
            self.en_cost = tf.add(self.en_cost, l2_loss)
            # cox part
            # Obtain T and E from self.Y
            # NOTE: negtive value means E = 0
            Y_c = tf.squeeze(self.Y)
            Y_hat_c = tf.squeeze(self.Y_hat)
            Y_label_T = tf.abs(Y_c)
            Y_label_E = tf.cast(tf.greater(Y_c, 0), dtype=tf.float32)
            Obs = tf.reduce_sum(Y_label_E)
            Y_hat_hr = tf.exp(Y_hat_c)
            Y_hat_cumsum = tf.log(tf.cumsum(Y_hat_hr))
            
            # Start Computation of Loss function
            # Get Segment from T
            unique_values, segment_ids = tf.unique(Y_label_T)
            # Get Segment_max
            loss_s2_v = tf.segment_max(Y_hat_cumsum, segment_ids)
            # Get Segment_count
            loss_s2_count = tf.segment_sum(Y_label_E, segment_ids)
            # Compute S2
            loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
            # Compute S1
            loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
            # Compute Breslow Loss
            loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)

            self.loss = tf.add(loss_breslow, l2_loss)
            self.loss = self.loss * self.loss_alpha + self.en_cost*(1-self.loss_alpha)
    def _create_optimizer(self):
        """
        Define optimizer
        """
        # SGD Optimizer
        if self.config["optimizer"] == 'sgd':
            lr = tf.train.exponential_decay(
                self.config["learning_rate"],
                self.global_step,
                1,
                self.config["learning_rate_decay"]
            )
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=self.global_step)
        # Adam Optimizer
        elif self.config["optimizer"] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.loss, global_step=self.global_step)
        elif self.config["optimizer"] == 'rms':
            self.optimizer = tf.train.RMSPropOptimizer(self.config["learning_rate"]).minimize(self.loss, global_step=self.global_step)     
        else:
            raise NotImplementedError('Optimizer not recognized')

    def build_graph(self):
        """Build graph of DeepCox
        """
        ###auto encoder
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        ###cox net work
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self.sess = tf.Session(config=tfconfig)

    def close_session(self):
        self.sess.close()
        tf.reset_default_graph()
        #print("Current session closed.")

    def train(self, data_X, data_y,valid_x,valid_y,test_x,test_y,num_steps, num_skip_steps=1,
              load_model="", save_model="", plot=False, silent=False):
        feed_data = {
            self.dropout: self.en_config["dropout"],
            self.training: True,
            self.loss_alpha: self.en_config["loss_alpha2"],
            self.X: data_X,
            self.Y: data_y
        }
        feed_data2 = {
            self.dropout: 0.0,
            self.training: False,
            self.loss_alpha: self.en_config["loss_alpha2"],
            self.X: valid_x,
            self.Y: valid_y
        }
        feed_data3 = {
            self.dropout: 0.0,
            self.training: False,
            self.loss_alpha: self.en_config["loss_alpha2"],
            self.X: test_x,
            self.Y: test_y
        }

        # Session Running
        self.sess.run(tf.global_variables_initializer())
        if load_model != "":
            saver = tf.train.Saver()
            saver.restore(self.sess, load_model)

        # we use this to calculate late average loss in the last SKIP_STEP steps
        total_loss = 0.0
        # Get current global step
        initial_step = self.global_step.eval(session=self.sess)
        # Record evaluations during training
        # watch_list = {'loss': [], 'metrics': []}
        valid_Cindex_set = []
        test_Cindex_set = []
        for index in range(initial_step, initial_step + num_steps + 1):
            if index < self.en_config["warm_up_epoch"]:
                print("warm_up")
                feed_data = {
                    self.dropout: self.en_config['dropout'],
                    self.loss_alpha: self.en_config["loss_alpha"],
                    self.training: True,
                    self.X: data_X,
                    self.Y: data_y
                }
            else:
                feed_data = {
                    self.dropout: self.en_config['dropout'],
                    self.loss_alpha: self.en_config["loss_alpha2"],
                    self.training: True,
                    self.X: data_X,
                    self.Y: data_y
                }
            y_hat, loss_value, _ = self.sess.run([self.Y_hat, self.loss, self.optimizer], feed_dict=feed_data)
            y_valid_hat = self.sess.run(self.Y_hat, feed_dict=feed_data2)
            y_test_hat = self.sess.run(self.Y_hat, feed_dict=feed_data3)
            train_Cindex = concordance_index(data_y, -y_hat)
            valid_Cindex = concordance_index(valid_y, -y_valid_hat)
            test_Cindex= concordance_index(test_y, -y_test_hat)
            if  index % num_skip_steps == 0 :
                print("index : " + str(index))
                print("train Cindex : " + str(train_Cindex))
                print("valid Cindex : " + str(valid_Cindex))
                print("test Cindex : " + str(test_Cindex))
                valid_Cindex_set.append(valid_Cindex)
                test_Cindex_set.append(test_Cindex)
                # print("loss : " + str(self.sess.run(self.loss, feed_dict=feed_data)))
                # print("en loss : " + str(self.sess.run(self.en_cost, feed_dict=feed_data)))

        # we only save the final trained model
        if save_model != "":
            # defaults to saving all variables
            saver = tf.train.Saver()
            saver.save(self.sess, save_model)
        # plot learning curve
        if plot:
            plot_train_curve(watch_list['loss'], title="Loss function")
            plot_train_curve(watch_list['metrics'], title="Concordance index")

        return valid_Cindex_set, test_Cindex_set

    def predict(self, X, output_margin=True):
        """
        Predict log hazard ratio using trained model.

        Parameters
        ----------
        X : DataFrame
            Input data with covariate variables, shape of which is (n, input_nodes).
        output_margin: boolean
            If output_margin is set to True, then output of model is log hazard ratio.
            Otherwise the output is hazard ratio, i.e. exp(beta*x).

        Returns
        -------
        np.array
            Predicted log hazard ratio (or hazard ratio) of samples with shape of (n, 1). 

        Examples
        --------
        >>> # "array([[0.3], [1.88], [-0.1], ..., [0.98]])"
        >>> model.predict(test_X)
        """
        # we set dropout to 1.0 when making prediction
        log_hr = self.sess.run([self.Y_hat], feed_dict={self.X: X, self.dropout: 0.0,self.training: False})
        if output_margin:
            return log_hr[0]
        return np.exp(log_hr)


    def evals(self, data_X, data_y):
        """
        Evaluate labeled dataset using the CI metrics under current trained model.

        Parameters
        ----------
        data_X, data_y: DataFrame
            Covariates and labels of survival data. It's suggested that you utilize 
            `tfdeepsurv.datasets.survival_df` to obtain the DataFrame object.

        Returns
        -------
        float
            CI metrics on your dataset.

        Notes
        -----
        We use negtive hazard ratio as the score. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
        """
        preds = - self.predict(data_X)
        return concordance_index(data_y, preds)

    def predict_survival_function(self, X, plot=False):
        """
        Predict survival function of samples.

        Parameters
        ----------
        X: DataFrame
            Input data with covariate variables, shape of which is (n, input_nodes).
        plot: boolean
            Plot the estimated survival curve of samples.

        Returns
        -------
        DataFrame
            Predicted survival function of samples, shape of which is (n, #Time_Points).
            `Time_Points` indicates the time point that exists in the training data.
        """
        pred_hr = self.predict(X, output_margin=False)
        survf = pd.DataFrame(self.BSF.iloc[:, 0].values ** pred_hr, columns=self.BSF.index.values)
        
        # plot survival curve
        if plot:
            plot_surv_curve(survf)

        return survf
        