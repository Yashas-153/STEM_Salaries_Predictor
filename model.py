import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestRegressor

df_init = pd.read_csv("E:\Projects\Data Science Salary predictor\Dataset\Levels_Fyi_Salary_Data.csv")
df = df_init.drop(['otherdetails','cityid', 'dmaid',
              'rowNumber','otherdetails', 'cityid', 'dmaid',
       'rowNumber', 'Masters_Degree', 'Bachelors_Degree', 'Doctorate_Degree',
       'Highschool', 'Some_College', 'Race_Asian', 'Race_White',
       'Race_Two_Or_More', 'Race_Black', 'Race_Hispanic'],axis = 1)

def pre_processing_data(df):
    
    df.drop(["totalyearlycompensation","stockgrantvalue","bonus"],axis = 1,inplace = True)

    df = df[df.basesalary > 10 ]
    q98 = np.quantile(df.basesalary,0.98)
    df.loc[ df.basesalary >q98 , 'basesalary'] =q98

    df = df.fillna({"company":"Unknown","level":"Unknown","tag":"Unknown","gender":"Unknown","Race":"Unknown","Education":"Unknown"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["ts_year"] = df.timestamp.dt.year
    df["ts_month"] = df.timestamp.dt.month
    df["ts_day"] = df.timestamp.dt.day


    com_encode = df.groupby("company")["basesalary"].median().reset_index()
    com_encode.columns = ["company","com_encode"]
    df = df.merge(com_encode)

    lev_encode = df.groupby("level")["basesalary"].median().reset_index()
    lev_encode.columns = ["level","level_encode"]
    df = df.merge(lev_encode)


    df = df.join(pd.get_dummies(df["title"]))
    df = df.join(pd.get_dummies(df["Education"]))

    df["country"] = df["location"].str.split(",").apply(lambda x: x[-1][1:])
    df.loc[df.country.str.len()== 2,'country']= "US"

    common_locs = df.value_counts('location').head(10).reset_index().location.to_list()
    df = df.join(pd.get_dummies(df.loc[df.location.isin(common_locs), 'location'] ) )
    df[common_locs] = df[common_locs].fillna(0)

    df = df.join(pd.get_dummies(df.country)[['US', 'India', 'Canada', 'United Kingdom', 'Germany']])
    y = df["basesalary"]
    X = df.drop(["basesalary","company","location","tag","level","country","title","timestamp","Race","gender","Education","yearsatcompany"],
       axis =1)
    cols = X.columns
    return X, y, cols, com_encode, lev_encode
 
X, y, cols, com_encoded, lev_encoded = pre_processing_data(df)

RF_model= RandomForestRegressor(random_state=0)
RF_model.fit(X, y)
#creating picklle files

pickle.dump(RF_model,open('Models/RandomForestRegressor.pkl','wb'))

pickle.dump(cols,open("Models/Columns.pkl","wb"))
pickle.dump(com_encoded,open("Models/company_encoded.pkl","wb"))
pickle.dump(lev_encoded,open("Models/level_encoded.pkl","wb"))




