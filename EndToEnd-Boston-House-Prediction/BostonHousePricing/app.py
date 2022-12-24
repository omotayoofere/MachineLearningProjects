import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
from flask.templating import render_template_string
import numpy as np
import pandas as pd


app=Flask(__name__)

model=pickle.load(open('regmodel.pkl','rb'))

##Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data'] #jsonify the input data
    print(data) #prints the data for view
    #reshaping the input data for standardization; np.array allows reshaping to be possible
    #reshaping into 1,-1 allows data to be treated point-to-point
    print(np.array(list(data.values())).reshape(1,-1)) 
    #pass reshaped data into scaling model
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1)) #
    #passing scaled data into regression model
    output=regmodel.predict(new_data)
    #printing output data
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()] #getting values from form
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=='__main__':
    app.run(debug=True)

#Procfile gives heroku the commands for the app to run on start-up
#Gunicorn--python http server wsgi applications. Allows to run python apps currently by running multiple proxies. 
#Distributes entire requests from multiple instances.