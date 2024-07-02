import os
import sys
from dataclasses import dataclass

# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import Custom_exception
from src.logger import logging
# from src.utils import save_object
from src.utils import perform_grid_search_cv, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("./artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig = ModelTrainerConfig()
    
    def initiateModelTrainer(self, train, test):

        try:
            logging.info("Splitting Training and Test Input Data...")
            X_train, y_train, X_test, y_test = (train.iloc[:,:-1],
                                                train.iloc[:,-1],
                                                test.iloc[:,:-1],
                                                test.iloc[:,-1])
            
            models = {"RF": RandomForestRegressor(),
                      "Ridge": Ridge(),
                      "DTR": DecisionTreeRegressor()}
            
            

            params_grid = {
                "RF":{'n_estimators': [50, 100, 120, 150, 200],
                      'max_depth': [5, 10, 15, 20]},
                "Ridge": {'alpha': [1, 2, 4, 6, 8]},
                "DTR": {'max_depth': [5, 10, 15, 20]}
                        }

            logging.info("Testing Different Model Performances...")

            estimator, modelname, score = perform_grid_search_cv(models, params_grid, X_test, y_test)
            logging.info(f"Best Score Obtained: {score}")

            if score<0.5:
                raise Custom_exception("No Best Model Found!!!", sys)
                

            logging.info(f"Best Model is {modelname}")

            ## Fitting the model

            estimator.fit(X_train, y_train)

            logging.info("Saving the Model")
            save_object(file_path = self.modeltrainerconfig.trained_model_file_path, 
                        obj = estimator)
            
            logging.info("Model Saved Successfully...")

            ## Model Prediction
            r_square = r2_score(y_true = y_test, y_pred = estimator.predict(X_test))

            logging.info(f"Models r2 score on test data: {r_square}")
            return r_square


        except Exception as e:
            raise Custom_exception(e, sys)




