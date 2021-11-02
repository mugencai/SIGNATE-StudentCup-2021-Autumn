class utils():
    def __init__(self):
        year_month = self.year_month
        cv_score = self.cv_score

    def YearMonth(year_month):
        if year_month == "201501":
            train_year_month, valid_year_month = "201411", "201412"
        elif year_month == "201502":
            train_year_month, valid_year_month = "201412", "201501"
        else:
            train_year_month, valid_year_month = str(int(year_month) - 2), str(int(year_month) - 1)
        test_year_month = year_month
        print("-----------------------------")
        print("Train: 201309 ～ " + train_year_month +" TrainSet" )
        print("Valid: " + valid_year_month + " TrainSet")
        print("Test:  " + test_year_month  + " TestSet")


    def CV(cv_score):
        print("------------ " + " END" + " ------------") 
        cv_score = float(sum(cv_score))/len(cv_score) 
        print ("CV: ", cv_score)  

    def get_yearmonth_list():
        _ = []
        for x in range(4):
            year_month = str(201409 + x)
            _.append(year_month)
        for x in range(1,9):
            year_month = str(201500 + x)
            _.append(year_month)
        return _