import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import csv


def load_data(fea_filename):
    data_fea = pd.read_csv(fea_filename)
    line_delete_set = pd.read_csv("line_delete_inter.csv")
    line_delete_set = [int(x) for x in line_delete_set]
    data_fea = data_fea.drop(line_delete_set, axis=0)
    headers = data_fea.columns.values.tolist()
    headers = headers[2:]
    headers = np.array(headers)



    data_fea = data_fea.drop('GeneSymbol', axis=1)
    data_fea = data_fea.drop('Platform', axis=1)

    time = data_fea.iloc[0,:].tolist()
    time = np.array(time)
    status = data_fea.iloc[1, :].tolist()
    status = np.array(status)
    data_fea = data_fea[2:]  ##delete label

    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_fea = data_fea.values
    data_time = time.reshape(-1, 1)
    return data_fea, data_time, headers



def sort_surv_data(X, Y):
    T = -np.abs(np.squeeze(np.array(Y)))
    sorted_idx = np.argsort(T)
    return X[sorted_idx], Y[sorted_idx]

def split_data(spilt_seed, fea, label,headers, nfold = 5, fold_num = 0):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
    ###对齐输入

    label_flat = label.flatten()
    censor_index = np.where(label_flat < 0)
    no_cen_index = np.where(label_flat >= 0)
    censor = label[[tuple(censor_index)]]
    censor_fea = fea[[tuple(censor_index)]]
    censor = censor[0]
    censor_fea = censor_fea[0]
    headers_censor = headers[[tuple(censor_index)]][0]

    no_cen = label[[tuple(no_cen_index)]]
    nocen_fea = fea[[tuple(no_cen_index)]]
    no_cen = no_cen[0]
    nocen_fea = nocen_fea[0]
    headers_nocensor = headers[[tuple(no_cen_index)]][0]
    num = 0
    for train_index, test_index in kf.split(censor_fea):
        train_X1 = censor_fea[train_index]
        train_Y1 = censor[train_index]
        test_X1 = censor_fea[test_index]
        test_Y1 = censor[test_index]
        headers_train1 = headers_censor[train_index]
        headers_test1 = headers_censor[test_index]
        if num == fold_num:
            break
        num +=1

    num = 0
    for train_index, test_index in kf.split(nocen_fea):
        train_X2 = nocen_fea[train_index]
        train_Y2 = no_cen[train_index]
        test_X2 = nocen_fea[test_index]
        test_Y2 = no_cen[test_index]
        headers_train2 = headers_nocensor[train_index]
        headers_test2 = headers_nocensor[test_index]
        if num == fold_num:
            break
        num +=1


    train_X = np.vstack((train_X1, train_X2))
    train_Y = np.vstack((train_Y1, train_Y2))
    test_X = np.vstack((test_X1, test_X2))
    test_Y = np.vstack((test_Y1, test_Y2))
    headers_train1 = headers_train1.tolist()
    headers_train2 = headers_train2.tolist()
    headers_test1 = headers_test1.tolist()
    headers_test2 = headers_test2.tolist()
    headers_train = headers_train1+headers_train2
    headers_test = headers_test1+headers_test2
    return headers_train,headers_test


# filenames = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirc", "lgg", "lihc", "luad", "lusc", "meso", "paad",
#              "sarc", "skcm"]
# filenames = ["esca", "hnsc", "kirc", "lgg", "lihc", "luad", "lusc", "meso", "paad",
#              "sarc", "skcm"]
filenames = ["coad"]

spilt_seed=2019
for f in filenames:
    fea_filename = "nozero/" + f + ".csv"
    ori_X, ori_Y, headers = load_data(fea_filename=fea_filename)
    ori_X = np.transpose(ori_X)
    headers_train, headers_test = split_data(spilt_seed, fea=ori_X, label=ori_Y,headers=headers, nfold=5,
                                                  fold_num=0)  ##get test

    line_delete_set = pd.read_csv("line_delete_inter.csv")
    line_delete_set = [int(x) for x in line_delete_set]
    dataset_fea = pd.read_csv(fea_filename)
    dataset_fea = dataset_fea.drop(headers_test, axis=1)
    dataset_fea = dataset_fea.drop(line_delete_set, axis=0)
    dataset_fea['GeneSymbol'] = dataset_fea['GeneSymbol'] + dataset_fea['Platform']
    dataset_fea = dataset_fea.drop('Platform', axis=1)
    dataset_fea.to_csv("baseline/" + f + "train.csv", index=False)

    dataset_fea2 = pd.read_csv(fea_filename)
    dataset_fea2 = dataset_fea2.drop(headers_train, axis=1)
    dataset_fea2 = dataset_fea2.drop(line_delete_set, axis=0)
    dataset_fea2['GeneSymbol'] = dataset_fea2['GeneSymbol'] + dataset_fea2['Platform']
    dataset_fea2 = dataset_fea2.drop('Platform', axis=1)
    dataset_fea2.to_csv("baseline/" + f + "test.csv", index=False)

