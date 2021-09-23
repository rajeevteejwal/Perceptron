import os
import joblib
import numpy as np
import pandas as pd
import logging
from utils.model import perceptron

def prepare_data(df):
    """[summary]

    Args:
        df ([DataFrame]): [description]
    """
    logging.info(f"Preparing training data")
    x = df.drop("y", axis=1)
    y = df["y"]
    return x, y

def save_model(model,file_name):
    model_dir = "models"
    os.makedirs(model_dir,exist_ok=True)
    file_path = os.path.join(model_dir,file_name)
    joblib.dump(model, file_path)

def get_model(file_name):
    model_dir = "models"
    model_path = os.path.join(model_dir,file_name)
    model = joblib.load(model_path)
    return model
