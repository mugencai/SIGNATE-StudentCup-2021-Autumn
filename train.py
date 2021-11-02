import lightgbm as lgb

cv_score = []

def train(traindata, testdata, year_month):
    if year_month == "201501":
        valid_split = traindata.loc[traindata["yearmonth"] == "201412"]
        train_split = traindata.loc[traindata["yearmonth"] < "201412"]
    else:
        valid_split = traindata.loc[traindata["yearmonth"] == str(int(year_month) - 1)]
        train_split = traindata.loc[traindata["yearmonth"] < str(int(year_month) - 1)]
    test_split = testdata.loc[testdata["yearmonth"] == year_month]
    
    train_split = train_split.drop(["date", "id", "predict"], axis=1)
    valid_split = valid_split.drop(["date", "id", "predict"], axis=1)
    print("train's shape: " + str(train_split.shape))
    print("valid's shape: " + str(valid_split.shape))

    
    # training
    X_train, X_test = train_split.drop(["yearmonth", "bikes_available"], axis=1), valid_split.drop(["yearmonth", "bikes_available"], axis=1)
    Y_train, Y_test = train_split['bikes_available'], valid_split['bikes_available']
    
    lgb_train=lgb.Dataset(X_train,Y_train)
    lgb_eval=lgb.Dataset(X_test,Y_test)
    params={
        'boosting': 'gbdt',
        'objective':'regression',
        'metric':'rmse',
        'learning_rate':0.01,
        'num_leaves': 191,         
        'device': 'gpu',}

    evals_result = {}
    early_stopping_rounds = 100
    model=lgb.train(params, lgb_train, 
                valid_names = ["train", "valid"],
                valid_sets=[lgb_train,lgb_eval],
                evals_result=evals_result, #dict{}でmetricを保存
                verbose_eval=500,  
                num_boost_round=3000, 
                early_stopping_rounds=early_stopping_rounds,
               )
    
    global cv_score
    if early_stopping_rounds == 0:
        cv_score.append(evals_result["valid"]["rmse"][-1])
    else:
        cv_score.append(evals_result["valid"]["rmse"][model.best_iteration - 1])
    
    return test_split, model, cv_score
