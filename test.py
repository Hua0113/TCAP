


from run_test import run_func
from dataset import get_dataset
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cindex_save = "result"   ##save Cindex in this file
prediction_save="pred"  ##save risk value in this file


encoder_config = {
    "n_hidden_1": 2000,  # first layer nodes num
    "n_hidden_2": 500,   # second layer nodes num
    # "L2_reg": 0.0001,
    # "L1_reg": 0.0001,
    "L2_reg": 1e-5,
    "L1_reg": 1e-5,
    "loss_alpha": 0.0,   # use for encoder warm up
    "loss_alpha2": 1,  # rate for encoder loss function
    "dropout": 0,
    "warm_up_epoch": 0
}
nn_config = {
    "learning_rate": 0.005,
    "learning_rate_decay":1,
    "num_steps":500,
    "pre_num_steps":0,
    "optimizer": 'sgd',
    "seed": 1,
    "spilt_seed":2020,
    "node_num":16        # third layer nodes num
}



is_pretrain = False #pretrain model
is_load = False    #load pretrain model
'''
    if you want to use transfer learning, set is_pretrain = True,
    if you have pretrained model file, set is_load=True, and there will not pretrian model.
    if you do not want to use transfer learning, just set is_pretrain = False, no matter what is_load set.
'''




### for ablation experiment
ablation = "no"
# ablation = "iscopyNumber"
# ablation = "isgeneExp"
# ablation = "ismethylation"
# ablation = "ismiRNAExp"
# ablation = "excopyNumber"
# ablation = "exgeneExp"
# ablation = "exmethylation"
# ablation = "exmiRNAExp"

save_idf = "test"     #identify for model
save_idf += ablation


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' New folder success')
        return True
    else:
        print(path + ' folder success')
        return False

mkdir(cindex_save)
mkdir(prediction_save)






fea_file_set = []
fea_file_set = get_dataset()




# hidden_l1list = [100,  50]
# hidden_l2list = [50, 40, 20]
# node_numlist = [10, 5]
# # learning_rate_list = [0.005]
# learning_rate_list = [0.01, 0.001]

hidden_l1list = [2000]
hidden_l2list = [200]
node_numlist = [50]
learning_rate_list = [0.00005]


times = 0
for h1 in hidden_l1list:
    for h2 in hidden_l2list:
        for node_num in node_numlist:
            for lr in learning_rate_list:
                time_file_set = []
                train_list_set = []
                test_list_set = []
                encoder_config["n_hidden_1"] = h1
                encoder_config["n_hidden_2"] = h2
                nn_config["node_num"]= node_num
                nn_config["learning_rate"]=lr
                num = 0
                save_idf_arguments = str(times) + save_idf
                for f_index in range(len(fea_file_set)):
                    num += 1
                    fea_filename = fea_file_set[f_index]
                    print("file name is : ")
                    print(fea_filename)
                    train_list_set.append(fea_filename)
                    test_list_set.append(fea_filename)
                    print(
                        "set up : " + " loss alpha2 : " + str(encoder_config["loss_alpha2"]) + " learning rate :" + str(
                            nn_config["learning_rate"]) + " num_steps: " + str(
                            nn_config["num_steps"]) + "seed : " + str(nn_config["spilt_seed"]))
                    train_s, test_s = run_func(ablation, save_idf_arguments, is_pretrain, is_load, prediction_save,
                                               encoder_config, nn_config, fea_filename)
                    train_list_set += train_s
                    test_list_set += test_s
                    is_load = True  ### load pretrained model

                # save cindex , epoch gap set in run.py
                with open(cindex_save + "/" + save_idf_arguments + "v.csv", "w",
                          ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(train_list_set)
                with open(cindex_save + "/" + save_idf_arguments + "t.csv", "w",
                          ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(test_list_set)
                times +=1

# num = 0
# for f_index in range(len(fea_file_set)):
#     num+=1
#     fea_filename = fea_file_set[f_index]
#     print("file name is : ")
#     print(fea_filename)
#     train_list_set.append(fea_filename)
#     test_list_set.append(fea_filename)
#     print("set up : " + " loss alpha2 : " + str(encoder_config["loss_alpha2"]) + " learning rate :" + str(
#         nn_config["learning_rate"]) + " num_steps: " + str(nn_config["num_steps"]) + "seed : " + str(nn_config["spilt_seed"]))
#     train_s, test_s = run_func(ablation, save_idf, is_pretrain, is_load, prediction_save, encoder_config, nn_config, fea_filename)
#     train_list_set+=train_s
#     test_list_set+=test_s
#     is_load = True   ### load pretrained model
#
#
# #save cindex , epoch gap set in run.py
# with open(cindex_save+"/"+save_idf+"v.csv", "w",
#           newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(train_list_set)
# with open(cindex_save+"/"+save_idf+"t.csv", "w",
#           newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(test_list_set)










