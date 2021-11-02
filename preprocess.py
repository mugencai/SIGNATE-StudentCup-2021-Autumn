import os
import sys
import pandas
import pandas as pd
import numpy as np
from pandas import Series, DataFrame



import warnings
warnings.simplefilter('ignore')


pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500) 


# pip install lightgbm --install-option=--gpu

status = pd.read_csv('status.csv') 
status["date"] = pd.to_datetime(status[["year", "month", "day"]])

station = pd.read_csv("station.csv")
station = station.drop(["installation_date",], axis=1)

weather = pd.read_csv('weather.csv') 
weather['date'] = pd.to_datetime(weather['date'],format="%Y-%m-%d")
weather = weather.drop(["events",                                     
#                         "max_temperature","min_temperature","mean_temperature",
#                         "max_dew_point","min_dew_point","mean_dew_point",
#                         "max_humidity","min_humidity","mean_humidity",
#                         "max_sea_level_pressure","mean_sea_level_pressure","min_sea_level_pressure",
#                         "max_visibility","min_visibility",
#                         "max_wind_Speed","cloud_cover","wind_dir_degrees",                                            
#                         "mean_visibility",
#                         "mean_wind_speed",                       
#                         "precipitation",                        
                       ], axis=1)


#曜日を追加する
def get_week(dt):
    w_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return(w_list[dt.weekday()])
status["weekday"] = status["date"].apply(get_week)


#"date"でstatus,station,weatherをmergeする
_ = pd.merge(status, station, on = "station_id")
_ = pd.merge(_, weather, on = "date")


#"city"と"weekday"をone-hot化
_ = pd.get_dummies(_, dummy_na=False) 


# "train"と"test"を区分
train = _.loc[_["predict"] == 0]
test = _.loc[_["predict"] == 1]


train.isna().any(axis=0)
test.isna().any(axis=0)

#trainの欠損値を削除
train = train.dropna(subset = ["bikes_available"])
train.shape

#"yearmonth"を追加する
def add_yearmonth(df):
    df1 = df[df['month'] < 10].copy()
    df2 = df[df['month'] >= 10].copy()
    df1['yearmonth'] = df1['year'].astype(str) + ('0' + df1['month'].astype(str))
    df2['yearmonth'] = df2['year'].astype(str) + df2['month'].astype(str)
    return pd.concat([df1, df2])

traindata = add_yearmonth(train)
testdata = add_yearmonth(test)


traindata.head(2)
testdata.head(2)

#年月をチェックする
def check_yearmonth(df):
    groupby = df.groupby("yearmonth").apply(lambda x: len(x)).reset_index()
    groupby.columns = ["yearmonth","size"]
    groupby = groupby.sort_values("yearmonth", ascending = True)
    print("size_sum: " + str(groupby["size"].sum()) )
    return groupby

check_yearmonth(traindata)
check_yearmonth(testdata)



# df = pd.merge(train_groupby,test_groupby,on="yearmonth",how="outer")
# df.fillna(0)
# df.plot.barh(x="yearmonth",align='center',xlabel="", figsize=(6,8), stacked=True)
