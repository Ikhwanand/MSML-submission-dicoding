import pandas as pd 
import numpy as np 
import joblib


def preprocessing_dataset(dataset_path=None, dataframe=None):
    if dataset_path:
        df = pd.read_csv(dataset_path)
    else:
        df = dataframe
    model_preprocessing = joblib.load('../models/preprocessor.pkl')    
    return model_preprocessing.fit_transform(df)

