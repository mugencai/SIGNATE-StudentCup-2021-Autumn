import pandas as pd
import lightgbm as lgb

def predict(test_split,model):
    #predict
    #test_idで予測結果の順を確保
    test_id = pd.DataFrame(test_split["id"])
    test_id = test_id.reset_index()
    del test_id["index"]

    test_feature = test_split.drop(["id", "predict", "bikes_available", "yearmonth", "date",], axis=1)
    test_label = model.predict(test_feature, num_iteration=model.best_iteration)
    test_label = pd.DataFrame(test_label)

    test_label = pd.concat([test_id, test_label], axis=1)
    return test_label
