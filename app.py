from flask import Flask, request, render_template 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from src.pipeline.prediction_pipeline import PredictPipeline,CustomData

app = Flask(__name__)

# Route for homepage
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_data():
  if request.method == 'GET':
    return render_template('home.html')
  else:
    data=CustomData(
      gender=request.form.get('gender'),
      race_ethnicity=request.form.get('race_ethnicity'),
      parental_level_of_education=request.form.get('parental_level_of_education'),
      lunch=request.form.get('lunch'),
      test_prep_course=request.form.get('test_prep_course'),
      reading_score=float(request.form.get('reading_score')),
      writing_score=float(request.form.get('writing_score'))
    )

    pred_df = data.get_dataframe()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html',results = results[0])
  
if __name__ == '__main__':
  app.run(debug=True)

  

