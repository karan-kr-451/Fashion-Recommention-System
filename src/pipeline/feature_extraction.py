import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from src.exception import FashionException
from src import logging
from src.utils.utils import save_pickle

# Global model and session to avoid re-initialization
global_model = None
global_sess = None

def initialize_model():
    global global_model
    global global_sess
    if global_model is None:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        global_model = tf.keras.Sequential([
            base_model,
            GlobalMaxPooling2D()
        ])
        global_sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(global_sess)

def extract_features_batch(batch_filenames):
    initialize_model()
    images = [image.load_img(file, target_size=(224, 224)) for file in batch_filenames]
    img_arrays = [image.img_to_array(img) for img in images]
    expanded_img_arrays = np.array([np.expand_dims(img_array, axis=0) for img_array in img_arrays])
    preprocessed_imgs = preprocess_input(np.vstack(expanded_img_arrays))
    with global_sess.as_default():
        results = global_model.predict(preprocessed_imgs)
    normalized_results = [result.flatten() / norm(result) for result in results]
    return normalized_results

class FeatureExtractor:
    logging.info('Feature extraction start')
    try:
        def __init__(self):
            initialize_model()
            self.model = global_model

        def extract_features_from_dir(self, dir_path, batch_size=32):
            filenames = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
            feature_list = []
            with Pool(cpu_count()) as pool:
                for i in tqdm(range(0, len(filenames), batch_size)):
                    batch_filenames = filenames[i:i+batch_size]
                    batch_features = pool.apply_async(extract_features_batch, (batch_filenames,))
                    feature_list.extend(batch_features.get())
            return feature_list, filenames

    except Exception as e:
        raise FashionException(e, sys) from e

if __name__ == "__main__":
    try:
        extractor = FeatureExtractor()
        feature_list, filenames = extractor.extract_features_from_dir('D:\\Github Project\\Fashion Recommendation\\fashion-dataset\\images')
        
        # Correctly save the features and filenames
        save_pickle('features.pkl', feature_list)
        save_pickle('filenames.pkl', filenames)

    except Exception as e:
        raise FashionException(e, sys)