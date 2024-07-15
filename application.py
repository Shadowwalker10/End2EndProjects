import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for home page

@app.route("/")


def index():
    return render_template("index.html")


@app.route("/predictdata", methods = ["GET", "POST"])

def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data_handler = CustomData(request.form.get("gender"),
                                  request.form.get("race_ethnicity"),
                                  request.form.get("parental_level_of_education"),
                                  request.form.get("lunch"),
                                  request.form.get("test_preparation_course"),
                                  request.form.get("reading_score"),
                                  request.form.get("writing_score")
                                  )
        data_frame = data_handler.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_frame)
        return render_template("home.html", results = results[0])
    
if __name__=="__main__":
    app.run(host = "0.0.0.0")




