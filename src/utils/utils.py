import os
import sys
import pickle
from src import logging
from src.exception import FashionException

def save_pickle(file_path, obj):
    logging.info('Entered the save_pickle method of utils')
    try:
        directory = os.path.dirname(file_path)
        if directory:  # Check if directory is not None or empty
            os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info('Exited the save_pickle method of utils')
    except Exception as e:
        raise FashionException(e, sys)

def load_pickle(file_path: str):
    logging.info('Entered the load_pickle method of utils')
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise FashionException(e, sys)
