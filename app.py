import pandas as pd
import numpy as np 
import pickle
import os
from flask import Flask,request, render_template

app=Flask(__name__,template_folder="templates")

@app.route('/', methods=['GET'])
@app.route('/home')
def index():
	return render_template('home.html')
@app.route('/about', methods=['GET'])
def about():
	return render_template('home.html')
@app.route('/pred',methods=['GET'])
def page():
	return render_template('upload.html')
@app.route('/predict', methods=['GET','POST'])
def predict() :
	print("[INFO] loading model...")
	model = pickle.loads(open('fdemand.pkl', "rb").read())
	input_features = [float(x) for x in request.form.values()]
	features_value = [np.array(input_features)]
	print(features_value)

	features_name = ['homepage_featured', 'emailer_for_promotion','op_area','cuisine','city_code','region_code','category']
	prediction = model.predict(features_value)
	output=prediction[0]
	print(output)
	return render_template('upload.html', prediction_text=output)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=False)



