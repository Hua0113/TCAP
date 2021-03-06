import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

set = []
data=pd.read_csv("brcpred.csv", header=None)
list = data.values.tolist()
print(list[0])
print(list[1])
pred = np.array(list[1])
print(pred)
pred_median = np.median(pred)
print(pred_median)
for i in range(len(pred)):
    if(pred[i] <= pred_median):
        pred[i] = 0
    else:
        pred[i] = 1
print(pred)
set.append(list[0])
set.append(pred)
with open("brcapred_median.csv", "w",
          newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(set)