import numpy as np
import pandas as pd
import pickle
import os
import sys
from src.logger import logging
from src.exceptions import Custom_exception
from sklearn.model_selection import GridSearchCV

from src.exceptions import Custom_exception

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise Custom_exception(e, sys)
    

def perform_grid_search_cv(model_dict:dict, params_grid:dict, x_test, y_test):
    try:
        logging.info("Performing Grid Search CV...")
        best_model = None
        best_score = -float('inf')
        best_estimator = None

        results = [(name, 
                    GridSearchCV(estimator=model, 
                                param_grid=params_grid[name], 
                                scoring="r2", 
                                cv = 5, 
                                n_jobs=-1).fit(x_test, y_test))
                    for name, model in model_dict.items()
                    ]
        for name, grid_search in results:
            if grid_search.best_score_>best_score:
                best_score = grid_search.best_score_
                best_estimator = grid_search.best_estimator_
                best_model = name

        return best_estimator, best_model, best_score
    except Exception as e:
        raise Custom_exception(e, sys)
    


def compile_text(x):
    text = f"""
    gender: {x['gender']},
    race_ethnicity: {x['race_ethnicity']},
    parental_level_of_education: {x['parental_level_of_education']},
    lunch: {x['lunch']},
    test_preparation_course: {x['test_preparation_course']},
    reading_score: {x['reading_score']},
    writing_score: {x['writing_score']}
    """
    return text
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        return obj

    except Exception as e:
        raise Custom_exception(e,sys)
    


        