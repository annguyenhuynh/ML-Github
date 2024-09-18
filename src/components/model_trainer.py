import os 
import sys 
from dataclasses import dataclass 

from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 

from src.exception import CustomException 
from src.logger import logging

from src.utils import save_object, evaluate_models
 
@dataclass 
class ModelTrainerConfig:
  trained_model_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig() 

  def initiate_model_training(self,train_array, test_array):
    try:
      logging.info("Split train and test input")
      X_train,y_train,X_test,y_test =(
        train_array[:,:-1], 
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      ) 

      models = {
        "Linear Regresssion": LinearRegression(),
        "K-Neighbor Regressor": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor(),
        "AdaBoost Regressor": AdaBoostRegressor()
      }

      model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
  

      # Get the best model from dict
      best_model_score=max(sorted(model_report.values()))
      
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]

      best_model = models[best_model_name] 
      
      if best_model_score < 0.7:
        raise CustomException("No best model found")
      logging.info(f"Best model found on both testing and training dataset")

      save_object(file_path=self.model_trainer_config.trained_model_path,
                  obj=best_model)
      
      predicted = best_model.predict(X_test) 
      r2_square = r2_score(y_test,predicted) 
      return r2_square 
      
     

    except Exception as e:
      raise CustomException(e,sys)
      



