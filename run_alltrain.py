from tfdeepsurv import auto_cox
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from dataset import get_dataset
import csv

def sort_surv_data(X, Y):
    T = -np.abs(np.squeeze(np.array(Y)))
    sorted_idx = np.argsort(T)
    return X[sorted_idx], Y[sorted_idx]

def split_data(spilt_seed, fea, label,headers, nfold=5, fold_num=0):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    num = 0
    for train_index, test_index in kf.split(fea):
        train_X = fea[train_index]
        train_Y = label[train_index]
        train_headers = headers[train_index]
        test_X = fea[test_index]
        test_Y = label[test_index]
        test_headers=headers[test_index]
        if num == fold_num:
            print(num)
            break
        num += 1
    train_X, train_Y = sort_surv_data(train_X, train_Y)
    return train_X, train_Y, test_X, test_Y, train_headers, test_headers

#
# def split_data(spilt_seed, fea, label, nfold = 5, fold_num = 0):
#     kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
#     ###
#
#     label_flat = label.flatten()
#     censor_index = np.where(label_flat < 0)
#     no_cen_index = np.where(label_flat >= 0)
#     censor = label[[tuple(censor_index)]]
#     censor_fea = fea[[tuple(censor_index)]]
#     censor = censor[0]
#     censor_fea = censor_fea[0]
#     no_cen = label[[tuple(no_cen_index)]]
#     nocen_fea = fea[[tuple(no_cen_index)]]
#     no_cen = no_cen[0]
#     nocen_fea = nocen_fea[0]
#
#     num = 0
#     for train_index, test_index in kf.split(censor_fea):
#         train_X1 = censor_fea[train_index]
#         train_Y1 = censor[train_index]
#         test_X1 = censor_fea[test_index]
#         test_Y1 = censor[test_index]
#         if num == fold_num:
#             break
#         num +=1
#
#     num = 0
#     for train_index, test_index in kf.split(nocen_fea):
#         train_X2 = nocen_fea[train_index]
#         train_Y2 = no_cen[train_index]
#         test_X2 = nocen_fea[test_index]
#         test_Y2 = no_cen[test_index]
#         if num == fold_num:
#             break
#         num +=1
#
#
#     train_X = np.vstack((train_X1, train_X2))
#     train_Y = np.vstack((train_Y1, train_Y2))
#     test_X = np.vstack((test_X1, test_X2))
#     test_Y = np.vstack((test_Y1, test_Y2))
#
#     train_X,train_Y= sort_surv_data(train_X,train_Y)
#     return train_X, train_Y, test_X, test_Y

# def split_data(spilt_seed, fea, label, nfold = 5, fold_num = 0):
#     kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
#     ###
#
#     label_flat = label.flatten()
#     censor_index = np.where(label_flat < 0)
#     no_cen_index = np.where(label_flat >= 0)
#     censor = label[[tuple(censor_index)]]
#     censor_fea = fea[[tuple(censor_index)]]
#     censor = censor[0]
#     censor_fea = censor_fea[0]
#     no_cen = label[[tuple(no_cen_index)]]
#     nocen_fea = fea[[tuple(no_cen_index)]]
#     no_cen = no_cen[0]
#     nocen_fea = nocen_fea[0]
#
#     num = 0
#     for train_index, test_index in kf.split(censor_fea):
#         train_X1 = censor_fea[train_index]
#         train_Y1 = censor[train_index]
#         test_X1 = censor_fea[test_index]
#         test_Y1 = censor[test_index]
#         if num == fold_num:
#             break
#         num +=1
#
#     num = 0
#     for train_index, test_index in kf.split(nocen_fea):
#         train_X2 = nocen_fea[train_index]
#         train_Y2 = no_cen[train_index]
#         test_X2 = nocen_fea[test_index]
#         test_Y2 = no_cen[test_index]
#         if num == fold_num:
#             break
#         num +=1
#
#
#     train_X = np.vstack((train_X1, train_X2))
#     train_Y = np.vstack((train_Y1, train_Y2))
#     test_X = np.vstack((test_X1, test_X2))
#     test_Y = np.vstack((test_Y1, test_Y2))
#
#     train_X,train_Y= sort_surv_data(train_X2,train_Y2)
#     return train_X, train_Y, test_X, test_Y






def pretrain(ablation, is_load, spilt_seed, my_model, node_num, save_idf, nfold, fold_num, encoder_config, nn_config):
    if is_load == True:  ##load
        return
    else:
        fea_file_set = get_dataset()
        save_mod = "./model_save/model" + save_idf + str(node_num) + str(
            int(encoder_config["loss_alpha2"] * 10)) + "s" + str(fold_num) + ".ckpt"
        pretrainset_fea = None
        pretrainset_t = None
        test_fea = None
        test_t = None
        for index in range(len(fea_file_set)):
            ori_X, ori_Y,headers = load_data(ablation=ablation, fea_filename=fea_file_set[index])
            train_X = np.transpose(ori_X)
            train_Y = ori_Y
            # train_X, train_Y, test_X, test_Y,train_headers,test_headers = split_data(spilt_seed, fea=ori_X, label=ori_Y, headers=headers,nfold=nfold,
            #                                               fold_num=fold_num)  ##get test
            if type(pretrainset_fea) is np.ndarray:
                pretrainset_fea = np.vstack((pretrainset_fea, train_X))
                pretrainset_t = np.vstack((pretrainset_t, train_Y))
                test_fea = np.vstack((test_fea, train_X))
                test_t = np.vstack((test_t, train_Y))
            else:
                pretrainset_fea = train_X
                pretrainset_t = train_Y
                test_fea = train_X
                test_t = train_Y
        pretrainset_fea, pretrainset_t = sort_surv_data(pretrainset_fea, pretrainset_t)
        train_list, test_list = my_model.train(pretrainset_fea, pretrainset_t, pretrainset_fea, pretrainset_t,
                                               test_fea, test_t, nn_config["pre_num_steps"], num_skip_steps=50,
                                               save_model=save_mod, plot=False)
        my_train_Cindex = my_model.evals(pretrainset_fea, pretrainset_t)
        my_test_Cindex = my_model.evals(test_fea, test_t)
        print("pr: CI on training data:", my_train_Cindex)
        print("pr: CI on test data:", my_test_Cindex)


def load_data(ablation, fea_filename):
    data_fea = pd.read_csv(fea_filename)

    headers = data_fea.columns.values.tolist()
    headers = headers[2:]
    headers = np.array(headers)
    # line_delete_set = pd.read_csv("line_delete_inter.csv")
    # line_delete_set = [int(x) for x in line_delete_set]
    # data_fea = data_fea.drop(line_delete_set, axis=0)

    time = data_fea.iloc[0,:].tolist()[2:]
    time = np.array(time)

    # min_v = min(time)
    # max_v = max(time)
    # div = max_v - min_v
    # for j in range(len(time)):
    #     if div == 0:
    #         time[j] = 0
    #     else:
    #         time[j] = (time[j] - min_v) / div

    status = data_fea.iloc[1, :].tolist()[2:]
    status = np.array(status)

    data_fea = data_fea[2:]  ##delete label

    if ablation != "no":
        dataname = ablation[2:]
        if ablation[0:2] == "is":  ##only this data
            data_fea = data_fea[data_fea['Platform'].isin([dataname])]
        else:  ##ex this data
            data_fea = data_fea[~data_fea['Platform'].isin([dataname])]
    data_fea = data_fea.drop('GeneSymbol', axis=1)
    data_fea = data_fea.drop('Platform', axis=1)

    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_fea = data_fea.values
    data_time = time.reshape(-1, 1)
    return data_fea, data_time, headers

def run_func(ablation="no", save_idf="save", is_pretrain=True, is_load=False,prediction_save="pred", run_encoder_config={}, run_nn_config={},
             fea_filename="none"):
    nfold = 5
    valid_list_set = []
    test_list_set = []
    ori_X, ori_Y, headers = load_data(ablation=ablation, fea_filename=fea_filename)
    ori_X = np.transpose(ori_X)
    train_X, train_Y = sort_surv_data(ori_X, ori_Y)
    for i in range(1):
        # train_X, train_Y, test_X, test_Y, train_headers, test_headers = split_data(spilt_seed=run_nn_config["spilt_seed"], fea=ori_X, label=ori_Y,
        #                                               headers=headers,nfold=nfold, fold_num=i)
        #########
        encoder_config = run_encoder_config
        nn_config = run_nn_config
        encoder_config["n_input"] = train_X.shape[1]

        ############cox part###
        print("n_input : " + str(encoder_config["n_input"]))
        node_num = run_nn_config["node_num"]
        print('node_num : ')
        print(node_num)
        # hidden_layers_nodes = [50, node_num, 1]
        hidden_layers_nodes = [node_num, 1]
        # ESSENTIAL STEP: Pass arguments
        model = auto_cox(
            hidden_layers_nodes,
            encoder_config,
            nn_config
        )
        # ESSENTIAL STEP: Build Computation Graph
        model.build_graph()
        my_load_model = ""
        if is_pretrain == True:
            pretrain(ablation, is_load, run_nn_config["spilt_seed"], model, node_num, save_idf, nfold, i,
                     encoder_config, nn_config)
            my_load_model = "./model_save/model" + save_idf + str(node_num) + str(
                int(encoder_config["loss_alpha2"] * 10)) + "s" + str(i) + ".ckpt"
            # my_load_model = "./model_save/model" + "0alltrainno410s0" + ".ckpt"
        valid_list, test_list = model.train(
            train_X, train_Y, train_X, train_Y, train_X, train_Y,
            num_steps=nn_config["num_steps"],
            # num_steps=0,
            num_skip_steps=20,
            # load_model=my_load_model,
            load_model="./model_save_coad/model" + "ov" + ".ckpt",
            # save_model="./model_save/model" + "prcob.ckpt",
            # load_model="./model_save_coad/model" + "30alltrainno5010s0" + ".ckpt",
            # save_model="./model_save_coad/model" + "ov.ckpt",
            # save_model="./model_save_cesc/model" + save_idf + str(node_num) + str(int(encoder_config["loss_alpha2"] * 10)) + "s" + str(i) + ".ckpt",

            plot=False
        )

        train_Cindex = model.evals(ori_X, ori_Y)
        test_Cindex = model.evals(ori_X, ori_Y)
        print("CI on training data:", train_Cindex)
        print("CI on test data:", test_Cindex)
        ###write_result
        valid_list_set.append(valid_list)
        test_list_set.append(test_list)

        from lifelines.statistics import logrank_test ###
        status = np.ravel(ori_Y).tolist()
        for i in range(len(status)):
            if (status[i] < 0):
                status[i] = 0
            else:
                status[i] = 1
        time = np.ravel(np.maximum(ori_Y, -ori_Y))
        pred = model.predict(ori_X)
        pred_median = np.median(pred)
        pred_int = pred.tolist()
        for i in range(len(pred_int)):
            if (pred_int[i] <= pred_median):
                pred_int[i] = 0
            else:
                pred_int[i] = 1
        print("median is : " + str(pred_median))
        set = []

        test_headers = ["name"] + np.ravel(headers).tolist()
        status= ["status"] + status
        time = ["time"] + np.ravel(time).tolist()
        pred = ["pred"] + np.ravel(pred).tolist()
        pred_int = ["risk_class"] + pred_int

        set.append(test_headers)
        set.append(time)
        set.append(status)
        set.append(pred)
        set.append(pred_int)



        with open(prediction_save+"/"+ save_idf + "pred" + fea_filename[8:], "w",
                  ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(set)
        model.close_session()
        #####
    return valid_list_set, test_list_set


