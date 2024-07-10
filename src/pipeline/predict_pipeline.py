import sys
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from src.exceptions import Custom_exception
from src.utils import load_object, compile_text

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, dataframe:pd.DataFrame):
        try:
            imputer_path = "./artifact/imputer.pkl"
            sentence_transformer_model_path = "./artifact/sentence-transformer/paraphrase-MiniLM-L12-v2"
            pca_path = "./artifact/pca.pkl"
            predictor_path = "./artifact/model.pkl"

            ## Loading Pickle Files
            imputer = load_object(imputer_path)
            pca = load_object(pca_path)
            model = load_object(predictor_path)

            ## Loading Sentence Transformer
            if not os.path.exists(sentence_transformer_model_path):
                sentence_transformer = SentenceTransformer(sentence_transformer_model_path)
            else:
                sentence_transformer = SentenceTransformer(sentence_transformer_model_path)

            cols = dataframe.columns
            ## Imputing the data
            dataframe = pd.DataFrame(imputer.transform(dataframe), columns = cols)

            ## Applying Sentence Transformer
            restructured_dataframe = dataframe.apply(lambda x: compile_text(x), axis = 1).tolist()
            restructured_dataframe = sentence_transformer.encode(restructured_dataframe, 
                                                                 show_progress_bar = True,
                                                                 normalize_embeddings = True)
            dataframe = pd.DataFrame(restructured_dataframe)
            # print(dataframe.head())


            ## Applying PCA
            dataframe = pd.DataFrame(pca.transform(dataframe))
            result = model.predict(dataframe)
            return result
        
        except Exception as e:
            raise Custom_exception(e. sys)



class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float
                 ):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score



    
    def get_data_as_dataframe(self):
        try:
            custom_data = {"gender": [self.gender],
                           "race_ethnicity": [self.race_ethnicity],
                           "parental_level_of_education": [self.parental_level_of_education],
                           "lunch": [self.lunch],
                           "test_preparation_course": [self.test_preparation_course],
                           "reading_score": [self.reading_score],
                           "writing_score": [self.writing_score]
                           }
            return pd.DataFrame(custom_data)

        except Exception as e:
            raise Custom_exception(e, sys)
    