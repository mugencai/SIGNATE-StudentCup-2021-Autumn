import os
import sys
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import lightgbm as lgb

from train import train
from predict import predict
from utils import utils
from preprocess import traindata, testdata 


test_labels =[]
test_labels = pd.DataFrame(test_labels)

yearmonth_list = utils.get_yearmonth_list()

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


submission.to_csv("submission.csv",index=False, header=False)
