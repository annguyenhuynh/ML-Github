import sys
import pandas as pd 
from src.exception import CustomException  
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
      try:
        model_path = "artifact/model.pkl"
        preprocessor_path = "artifact/preprocessor.pkl"
        
        # Logging to trace the process
        print("Before loading model and preprocessor")
        
        # Load model and preprocessor
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        
        print("After loading model and preprocessor")
        
        # Log input features before transformation
        print(f"Input features before transformation:\n{features}")

        # Ensure preprocessing can handle unknown categories and check the transformation
        data_scaled = preprocessor.transform(features)

        # Log transformed data
        print(f"Transformed features after preprocessing:\n{data_scaled}")
        
        # Predict
        preds = model.predict(data_scaled)
        
        # Log predictions
        print(f"Predictions:\n{preds}")
        
        return preds

      except Exception as e:
          print(f"Error in prediction pipeline: {str(e)}")  # Log the actual error message
          raise CustomException(str(e), sys)
  
class CustomData:
  def __init__(self,
               gender:str,
               race_ethnicity:str,
               parental_level_of_education:str,
               lunch:str,
               test_prep_course: str,
               reading_score,
               writing_score):
    
    self.gender = gender
    self.race_ethnicity = race_ethnicity 
    self.parental_level_of_eduction=parental_level_of_education 
    self.lunch=lunch 
    self.test_prep_course=test_prep_course
    self.reading_score = reading_score 
    self.writing_score = writing_score 

  def get_dataframe(self):
    try:
      custom_data_input_dict = {
        "gender":[self.gender],
        "race/ethnicity":[self.race_ethnicity],
        "parental level of education":[self.parental_level_of_eduction],
        "lunch":[self.lunch],
        "test preparation course":[self.test_prep_course],
        "reading score":[self.reading_score],
        "writing score":[self.writing_score]
      
      }
      df = pd.DataFrame(custom_data_input_dict)
      for column in df.columns:
        df[column].fillna("Unknown" if df[column].dtype == object else df[column].mean(), inplace=True)
        
        return df
    
    except Exception as e:
      raise CustomException(e,sys)
