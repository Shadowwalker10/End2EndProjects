import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer
from src.exceptions import Custom_exception
from src.logger import logging
from src.utils import save_object, compile_text

@dataclass
class DataTransformationConfig:
    sentence_transformer_model : str = os.path.join("./artifact", "sentence-transformer", "paraphrase-MiniLM-L12-v2")
    pca_model : str = os.path.join("./artifact", "pca.pkl")
    imputer_model :str = os.path.join("./artifact", "imputer.pkl")
    embedded_train : str = os.path.join("./artifact", "embedded_train.csv")
    embedded_test : str = os.path.join("./artifact", "embedded_test.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_imputer_object(self):
        try:
            numerical_cols = ['reading_score', 'writing_score']
            categorical_cols = ['gender', 'race_ethnicity', 
                                'parental_level_of_education', 'lunch',
                                'test_preparation_course']
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                ]
            )

            cat_pipeline = Pipeline(steps = [
                ("imputer", SimpleImputer(strategy = "most_frequent"))
            ])

            logging.info("Null Values Imputed!!")

            imputer_preprocesser = ColumnTransformer([
                ("numerical_pipeline", num_pipeline, numerical_cols),
                ("categorical pipeline", cat_pipeline, categorical_cols)
            ])

            logging.info("Numerical and Categorical Pipeline Output Combined")
            return imputer_preprocesser
        

        except Exception as e:
            raise Custom_exception(e, sys)
        
    def apply_sentence_transformer(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, index_col=False).iloc[:,1:]
            test_df = pd.read_csv(test_path, index_col=False).iloc[:,1:]

            logging.info("Read the Train-Test Data!!")

            logging.info("Carrying Out Data Imputation....")

            imputer_object = self.get_imputer_object()

            target_col_name = "math_score"
            numerical_cols = ['reading_score', 'writing_score']

            input_train_features = train_df.drop(target_col_name, axis = 1)
            columns = input_train_features.columns

            target_feature_train_df = train_df[target_col_name]

            input_test_features = test_df.drop(target_col_name, axis = 1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying Imputer Object on Train and Test Features")
            input_train_features = imputer_object.fit_transform(input_train_features)
            input_test_features = imputer_object.transform(input_test_features)

            logging.info("Values Imputed Successfully...")

            ## Saving the imputer object
            logging.info("Saving the Imputer Object...")

            save_object(self.data_transformation_config.imputer_model, imputer_object)

            logging.info("Imputer Object Saved Sucessfully...")


            ##Applying Sentence Transformer and PCA
            ### Defining Function for Data Restructuring
            
            input_train_features = pd.DataFrame(input_train_features, columns=columns)

            input_test_features = pd.DataFrame(input_test_features, columns=columns)

            
            logging.info("Initiated Train-Test Compilation to List of Texts")
            restructured_train = input_train_features.apply(lambda x: compile_text(x), axis = 1).tolist()
            restructured_test =input_test_features.apply(lambda x: compile_text(x), axis = 1).tolist()
            
            logging.info("Completed Text Compilation...")

            logging.info("Loading Sentence Transformer Model...")
            model_path = self.data_transformation_config.sentence_transformer_model
            if not os.path.exists(model_path):
                logging.info(f"Path {model_path} not found. Downloading the model...")
                model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
                # model.save(model_path)
            else:
                model = SentenceTransformer(model_name_or_path = model_path)
            
            logging.info("Loaded Sentence Transformer Model...")
            train_output = model.encode(sentences  = restructured_train, 
                                        show_progress_bar = True,
                                        normalize_embeddings = True)
            test_output = model.encode(
                sentences = restructured_test,
                show_progress_bar = True,
                normalize_embeddings = True
            )

            # print("Transformer Output: ", train_output.shape)

            logging.info("Completed Encoding of Train Test Data Using Transformer...")
            df_train_embedded = pd.concat([pd.DataFrame(train_output), 
                                           train_df[target_col_name].reset_index(drop = True)], 
                                           axis = 1)
            df_test_embedded = pd.concat([pd.DataFrame(test_output), 
                                          test_df[target_col_name].reset_index(drop = True)],
                                          axis=1)

            df_train_embedded.to_csv(path_or_buf = self.data_transformation_config.embedded_train, index = False)
            df_test_embedded.to_csv(path_or_buf = self.data_transformation_config.embedded_test, index = False)
            # print("After addding target: ", df_train_embedded.shape)

        except Exception as e:
            raise Custom_exception(e, sys)


    def apply_pca(self):
        try:
            ## Load the embedded train test csv file
            embedded_train_df = pd.read_csv(self.data_transformation_config.embedded_train)
            embedded_test_df = pd.read_csv(self.data_transformation_config.embedded_test)
            # print("Loaded for pca: ", embedded_train_df.shape)
            train_target = embedded_train_df.iloc[:,-1]
            test_target = embedded_test_df.iloc[:,-1]

            ## Initialize PCA
            ### Based on Model-Training.ipynb : best number of pca components = 30
            
            pca = PCA(n_components = 30)
            logging.info("PCA Initialized...")
            # print("Taken for pca: ", embedded_train_df.iloc[:, :-1].shape)
            pca_train = pca.fit_transform(embedded_train_df.iloc[:,:-1])
            pca_test = pca.transform(embedded_test_df.iloc[:,:-1])

            logging.info("PCA Applied Successfully...")

            pca_train_df = pd.concat([pd.DataFrame(pca_train), 
                                      train_target.reset_index(drop = True)], 
                                      axis = 1)
            
            pca_test_df = pd.concat([pd.DataFrame(pca_test), 
                                     test_target.reset_index(drop = True)], 
                                     axis = 1)

            save_object(self.data_transformation_config.pca_model, pca)

            logging.info("PCA Model Saved Successfully...")

            return pca_train_df, pca_test_df


        except Exception as e:
            raise Custom_exception(e, sys)
