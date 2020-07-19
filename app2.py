# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:21:02 2020

@author: Sumit Keshav
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle as p
import pickle
import csv

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/make_predictions')
def make_predictions():
    return render_template('make_predictions.html')

@app.route('/enter_value')
def enter_value():
    return render_template('predict.html')

@app.route('/upload_csv')
def upload_csv():
    return render_template('uploadcsv.html')

@app.route('/Cloud_ICU_Model')
def Cloud_ICU_Model():
    return render_template('Cloud_ICU_Model.html')

@app.route('/predict_using_csv',methods=['POST','GET'])
def predict_using_csv():
    
    f = request.form['csvfile']
    data = []

    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data.append(row)
    
    count = 1
    predictionstrings = []
    predictionstate = []

    for d in data:

        form_value = [float(x) for x in d]
        feature_array = np.array(form_value).reshape(1,-1)
        std=p.load(open("scaler.p","rb"))
        #res = std.transform(form_value)
        res = np.array(form_value).reshape(1,-1)
        res = std.transform(res)
        model_prob = pickle.load(open("rf.pkl","rb"))
        model_reg = pickle.load(open("LoS_model.pkl","rb"))
        prediction_survival = model_prob.predict_proba(res)
        prediction_LoS = model_reg.predict(feature_array) # Need not apply scaling in this, since its Gradient Boosting
        output = '{0:.{1}f}'.format(prediction_survival[0][1],4)
        current_prediction = ""

        if output>str(0.311):
            current_prediction = 'Patient number {} is in CRITICAL STATE. Patient survives with a LOW probability of :{:.2f} %. The Length of Stay of the patient is: {} days'.format(count,(1.0-float(output))*100,int(prediction_LoS))
            predictionstate.append(True)
        else:
            current_prediction='Patient number {} is in SAFE STATE. Patient survives with a HIGH probability of: {:.2f} %. The Length of Stay of the patient is : {} days'.format(count,(1.0-float(output))*100,int(prediction_LoS))
            predictionstate.append(False)
        
        predictionstrings.append(current_prediction)
        count = count + 1

 
    return render_template('output_multi.html',pred = predictionstrings,predstate = predictionstate)


@app.route('/predict',methods=['POST','GET'])
def predict():
    
    form_value = [float(x) for x in request.form.values()]
    feature_array = np.array(form_value).reshape(1,-1)
    std=p.load(open("scaler.p","rb"))
    #res = std.transform(form_value)
    res = np.array(form_value).reshape(1,-1)
    res = std.transform(res)
    model_prob = pickle.load(open("rf.pkl","rb"))
    model_reg = pickle.load(open("LoS_model.pkl","rb"))
    prediction_survival = model_prob.predict_proba(res)
    prediction_LoS = model_reg.predict(feature_array) # Need not apply scaling in this, since its Gradient Boosting
    output = '{0:.{1}f}'.format(prediction_survival[0][1],4)
    
    if output>str(0.311):
        return render_template('output.html',pred = 'Patient is in CRITICAL STATE. Patient survives with a LOW probability of :{:.2f} %. The Length of Stay of the patient is: {} days'.format( (1.0-float(output))*100 , int(prediction_LoS) ), danger = True) 
    else:
        return render_template('output.html',pred = 'Patient is in SAFE STATE. Patient survives with a HIGH probability of: {:.2f} %. The Length of Stay of the patient is : {} days'.format((1.0-float(output))*100,int(prediction_LoS)), danger = False )
        
if __name__ == "__main__":
    app.run(debug=True)