# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:21:02 2020

@author: Sumit Keshav
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle as p

app = Flask(__name__)
model = p.load(open('rf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/view_records')
def view_records():
    return render_template('predict.html')

@app.route('/add_records')
def add_records():
    return render_template('predict.html')

@app.route('/make_predictions')
def make_predictions():
    return render_template('make_predictions.html')



@app.route('/predict',methods=['POST','GET'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    std=p.load(open("scaler.p","rb"))
    cols = pd.read_csv("master.csv").drop(columns=['Timestamp','RecordID']).columns
    
    features = np.array(int_features).reshape(1,-1)
    df = pd.DataFrame(features)
    df.columns = cols
    
    missing_vi = ['Glucose','Mg']
    
    
    
    res = std.transform(features)
    prediction = model.predict_proba(res)
    output = '{0:.{1}f}'.format(prediction[0][1],2)

    if output>str(0.311):
        return render_template('index2.html',pred = 'Patient is in CRITICAL STATE.\nPatient dies with a probability of :{}.\n Addital Vitals to take care of: {}'.format(output,missing_vi))
    else:
        return render_template('index2.html',pred = 'The patient can now safely survive according to the given vitals.\nPatient survives with a probability of: {}\n Addital Vitals to take care of: {}'.format(1-float(output),missing_vi))
        
if __name__ == "__main__":
    app.run(debug=True)