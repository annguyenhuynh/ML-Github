import os
import sys 
from src.exception import CustomException
from src.logger import logging 
import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass  
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation: 
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig() 
     
  def get_data_transformer(self):
    try:
      numerical_columns=['writing score', 'reading score'] 
      categorical_features = [
        'gender',
        'race/ethnicity',
        'parental level of education',
        'lunch',
        'test preparation course'
      ]

      numerical_pipeline = Pipeline(
        steps = [
          ("imputer", SimpleImputer(strategy="median")), #the data have outliers
          ("scaler", StandardScaler())
        ]
      )
      
      categorial_pipeline = Pipeline(
        steps = [
          ("imputer", SimpleImputer(strategy="most_frequent")),
          ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
          ("scaler", StandardScaler())
        ]
      )

      logging.info("Categorical columns: {categorical_features}")
      logging.info("Numerical columns: {numerical_columns}")

    

      preprocessor = ColumnTransformer(
        [
          ("numerical_pipeline", numerical_pipeline, numerical_columns ),
          ("categorical_pipeline", categorial_pipeline,categorical_features )
        ]
      )

      return preprocessor

    except Exception as e:
      raise CustomException(e,sys)
    
  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)

      logging.info("Read train and test data")

      logging.info("Obtain preprocessing oject")

      preprocessor_obj = self.get_data_transformer()

      target_column_name = "math score"
      numerical_columns=['writing score', 'reading score']  
      
      input_features_train_df=train_df.drop(columns=[target_column_name], axis=1)
      target_feature_train_df=train_df[target_column_name]

      input_features_test_df=test_df.drop(columns=[target_column_name], axis=1)
      target_feature_test_df=test_df[target_column_name]

      logging.info("Applying preprocessing object on train and test df")

      input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
      input_features_test_arr = preprocessor_obj.transform(input_features_test_df)

      train_arr = np.c_[input_features_train_arr,np.array(target_feature_train_df)] #np_c: concatenate arrays along the columns (2nd axis)
      test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]

      logging.info(f"Save preprocessing object")
      
      save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessor_obj
      )

      logging.info(f"Attempting to save pickle file at: {self.data_transformation_config.preprocessor_obj_file_path}")



      return(
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path
      )
      

    except Exception as e:
      raise CustomException(e, sys)
