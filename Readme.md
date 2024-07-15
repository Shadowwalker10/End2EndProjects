# Maths Score Prediction

This repository contains an end-to-end machine learning project aimed at predicting Maths scores based on various factors. The project makes use of sentence transformers for data encoding, PCA for dimensionality reduction, and hyperparameter tuning to find the best model parameters. Additionally, a Flask app is developed for serving the model predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Shadowwalker10/End2EndProjects
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Update the Sentence Transformer model:
    Ensure that the `./artifact/sentence-transformer/paraphrase-MiniLM-L12-v2` folder contains the necessary model tensors.

## Usage

There are two ways to use this project:

1. **Running the Flask App directly:**
    - Run the Flask app:
        ```bash
        python app.py
        ```
    - Open a new tab in your browser and navigate to `http://127.0.0.1:5000/predictdata`.

2. **Finding your own best model and then running the Flask app:**
    - Run the data ingestion and model training script:
        ```bash
        python data_ingestion.py
        ```
    - Run the Flask app:
        ```bash
        python app.py
        ```
    - Open a new tab in your browser and navigate to `http://127.0.0.1:5000/predictdata`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

