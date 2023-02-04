from flask import Flask, render_template, request
from datetime import date 
import pandas as pd
import numpy as np
import pickle

lev_encoded = pickle.load(open("Models/level_encoded.pkl","rb"))
cols = pickle.load(open("Models/Columns.pkl","rb"))
comp_encoded = pickle.load(open("Models/company_encoded.pkl","rb"))
model = pickle.load(open("Models/RandomForestRegressor.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    company = request.form['c_name']    
    title = request.form['title_name']
    location = request.form['loc_name']
    experience = request.form['yearsofexperience']
    Education = request.form['education']
    level = request.form['level']
    pred_df = pre_processing(company,title,location,experience,Education,level)
    prediction = model.predict(pred_df)
    return render_template('index.html', predict = "The predicted salary is ${0} per annum".format(prediction[0]))



def pre_processing(com,tit,loc,exp,edu,level):
    com = str.capitalize(com)
    loc = str.capitalize(loc)
    feat = []
    feat.append(exp)
    cur_date = date.today()
    cur_year = cur_date.year
    cur_mon = cur_date.month
    cur_day = cur_date.day
    feat.append(cur_year)
    feat.append(cur_mon)
    feat.append(cur_day)
    pred_df = pd.DataFrame(columns = cols)
    pred_df.loc[0,:] = 0
    pred_df.loc[0,tit] = 1
    pred_df.loc[0,edu] = 1
    pred_df.loc[0,"ts_year"] = cur_year
    pred_df.loc[0,"ts_month"] = cur_mon
    pred_df.loc[0,"ts_day"] = cur_day
    pred_df.loc[0,"yearsofexperience"] = exp
    pred_df.loc[0,"com_encode"] = comp_encoded[comp_encoded["company"] == com].values[0,1]
    pred_df.loc[0,"level_encode"] = lev_encoded[lev_encoded["level"] == level].values[0,1]
    print(pred_df)
    return pred_df
    




    