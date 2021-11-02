
from train import train
from predict import predict
from utils import utils

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from preprocess import traindata, testdata 

def get_yearmonth_list():
    _ = []
    for x in range(4):
        year_month = str(201409 + x)
        _.append(year_month)
    for x in range(1,9):
        year_month = str(201500 + x)
        _.append(year_month)
    return _


test_labels =[]
test_labels = pd.DataFrame(test_labels)

yearmonth_list = get_yearmonth_list()

while yearmonth_list != []:
    year_month = yearmonth_list.pop(0)
    utils.YearMonth(year_month)
    test_split, model, cv_score = train(traindata, testdata, year_month)
    test_label = predict(test_split, model)
    test_labels = pd.concat([test_labels, test_label])
    print(" ")
    
utils.CV(cv_score)


test_labels.columns = ["id", "pred"]
submission = pd.DataFrame(test_labels)
# submission.info()


submission.to_csv("submission.csv",index=False, header=False)