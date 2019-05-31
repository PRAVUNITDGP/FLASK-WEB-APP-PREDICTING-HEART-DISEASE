from flask import Flask ,render_template,url_for,request
from flask_material import Material


import pandas as pd
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)
Material(app)
@app.route('/')
    
def index():
    return render_template('index.html')

@app.route('/preview')
def preview():
    df = pd.read_csv("data/heart.csv")
    return render_template("preview.html",df_view = df)
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        model_choice=request.form['model_choice']

        sample_data=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        clean_data =[float(i) for i in sample_data]
        #reshaping 
        r1=np.array(clean_data).reshape(1,-1)

        #SELECTING MODELS FOR PREDICTION .... LOADING PREVIOUSLY RUN MODELS FROM OUR DATA DIRECTORY 


        if model_choice =='logitmodel':
            logitmodel = joblib.load("data/heartLR.pkl")
            predicted_result=logitmodel.predict(r1)
        elif model_choice =='knnmodel':
            logitmodel = joblib.load("data/heartKNN.pkl")
            predicted_result=logitmodel.predict(r1)
        elif model_choice =='svmodel':
            logitmodel = joblib.load("data/heartSVC.pkl")
            predicted_result=logitmodel.predict(r1)
        elif model_choice =='naivebaise':
            logitmodel = joblib.load("data/heartNB.pkl")
            predicted_result=logitmodel.predict(r1)
        elif model_choice =='decesiontree':
            logitmodel = joblib.load("data/heartDtree.pkl")
            predicted_result=logitmodel.predict(r1)
        #IF NOTHING WILL BE CHOSEN THEN OUR DEFAULT MODEL LOGISTIC REGRESSION MODEL WILL RUN     
        else:
            logitmodel = joblib.load("data/heartLR.pkl")
            predicted_result=logitmodel.predict(r1)
                

    return render_template('index.html',age=age,sex=sex,cp=cp,trestbps=trestbps,chol=chol,fbs=fbs,restecg=restecg,thalach=thalach,exang=exang,oldpeak=oldpeak,slope=slope,ca=ca,thal=thal,clean_data=clean_data,predicted_result=predicted_result)     

if __name__ == '__main__':
    app.run(debug=True)
 