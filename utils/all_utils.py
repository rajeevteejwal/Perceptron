import os
import joblib
import numpy as np
import pandas as pd
import logging

def prepareData(df):
    """[summary]

    Args:
        df ([DataFrame]): [description]
    """
    logging.info(f"Preparing training data")
    x = df.drop("y", axis=1)
    y = df["y"]
    return x, y