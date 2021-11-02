import pandas as pd

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














