import os
import sys
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

import warnings
warnings.simplefilter('ignore')

pip install lightgbm --install-option=--gpu


status = pd.read_csv('status.csv') 
status["date"] = pd.to_datetime(status[["year", "month", "day"]])


station = pd.read_csv("station.csv")
station = station.drop(["installation_date",], axis=1)

weather = pd.read_csv('weather.csv') 
weather['date'] = pd.to_datetime(weather['date'],format="%Y-%m-%d")
weather = weather.drop(["events",                                     
                        "max_temperature","min_temperature","mean_temperature",
                        "max_dew_point","min_dew_point","mean_dew_point",
                        "max_humidity","min_humidity","mean_humidity",
                        "max_sea_level_pressure","mean_sea_level_pressure","min_sea_level_pressure",
                        "max_visibility","min_visibility",
                        "max_wind_Speed",
                        "cloud_cover",
                        "wind_dir_degrees",
#                         "mean_visibility",
#                         "mean_wind_speed",
#                         "precipitation",
                       ], axis=1)
                       
#通过date新增“weekday”列
#“weekday”列后续处理参照“チュートリアル第二弾”
#曜日を追加するための関数を定義
def get_week(dt):
    w_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return(w_list[dt.weekday()])
status["weekday"] = status["date"].apply(get_week)

status.head(2)

_ = pd.merge(status, station, on = "station_id")
_ = pd.merge(_, weather, on = "date")

train = _.loc[_["predict"] == 0]
test = _.loc[_["predict"] == 1]

#删掉train里头的缺失值
train = train.dropna(subset = ["bikes_available"])
train.shape


#给train，test添加"yearmonth"新列以便后续拆解小test
df_m_s = train[train['month'] < 10].copy()
df_m_b = train[train['month'] >= 10].copy()
df_m_s['yearmonth'] = df_m_s['year'].astype(str) + ('0' + df_m_s['month'].astype(str))
df_m_b['yearmonth'] = df_m_b['year'].astype(str) + df_m_b['month'].astype(str)
train = pd.concat([df_m_s, df_m_b])

test = test.copy()
df_m_s = test[test['month'] < 10].copy()
df_m_b = test[test['month'] >= 10].copy()
df_m_s['yearmonth'] = df_m_s['year'].astype(str) + ('0' + df_m_s['month'].astype(str))
df_m_b['yearmonth'] = df_m_b['year'].astype(str) + df_m_b['month'].astype(str)
test = pd.concat([df_m_s, df_m_b])

#train年-月groupby
groupby = train.groupby("yearmonth").apply(lambda x: len(x)).reset_index()
groupby.columns = ["train_yearmonth","numbers"]
groupby = groupby.sort_values("train_yearmonth", ascending = True)
print(groupby["numbers"].sum())

#test年-月groupby
groupby = test.groupby("yearmonth").apply(lambda x: len(x)).reset_index()
groupby.columns = ["test_yearmonth","numbers"]
print(groupby["numbers"].sum())



