import sys
import os
from tempfile import TemporaryDirectory
import xlearn as xl
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from reco_utils.common.constants import SEED
from reco_utils.common.timer import Timer
from reco_utils.dataset.download_utils import maybe_download, unzip_file
from reco_utils.tuning.parameter_sweep import generate_param_grid
from reco_utils.dataset.pandas_df_utils import LibffmConverter

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from itertools import compress
import re

# read original cretio data
test = pd.read_csv("cretio_small.csv")
test = test.drop(['Id'], axis = 1)
labels = pd.DataFrame(test['Label'])
test = test.drop(['Label'], axis = 1)

num_features = [re.search("I", x) for x in test.columns]
num_features = list(compress(test.columns, num_features))
cat_features = [re.search("C", x) for x in test.columns]
cat_features = list(compress(test.columns, cat_features))

# TODO: sample data

# make sure index is OK.
#out = out.reset_index()

# sample data randomly in training and evaluation data sets.
#train = out.sample(frac=frac, random_state=random_state)
#eval = out.drop(train.index)

# preprocessing pipeline - impute NA
preprocessor = ColumnTransformer(
    transformers=[
        ('impute_num', SimpleImputer(strategy='median'), num_features),
        ('impute_cat', SimpleImputer(strategy='constant', fill_value = "NA"), cat_features)])

preprocessor.fit(test)

# impute values
features = preprocessor.transform(test)
features = pd.DataFrame(data = features)

out = pd.concat([labels, features], axis = 1)

# convert to (libsvm)libffm format
converter = LibffmConverter().fit(out, col_rating='Label')
out = converter.transform(out)

path_trn = "criteo/cretio_trn"
out.to_csv(path_trn, sep=' ', index=False, header=False)

# train FM
fm_model = xl.create_fm()

fm_model.setTrain(path_trn)     # Set the path of training dataset
fm_model.setValidate(path_trn)  # Set the path of validation dataset

param = {"task":"binary", 
         "lr": 0.2, 
         "lambda": 0.02, 
         "metric": 'auc',
         "epoch": 10,
         "opt": 'sgd'}

fm_model.fit(param, "./model_output")

#
#
#print("System version: {}".format(sys.version))
#print("Xlearn version: {}".format(xl.__version__))
#
#
#
#
#
#
#
## original data
#
#df_feature_original = pd.DataFrame({
#    'rating': [1, 0, 0, 1, 1],
#    'field1': ['xxx1', 'xxx2', 'xxx4', 'xxx4', 'xxx4'],
#    'field2': [3, 4, 5, 6, 7],
#    'field3': [1.0, 2.0, 3.0, 4.0, 5.0],
#    'field4': ['1', '2', '3', '4', '5']
#})
#
#converter = LibffmConverter().fit(df_feature_original, col_rating='rating')
#df_out = converter.transform(df_feature_original)
#df_out
#
#print('There are in total {0} fields and {1} features.'.format(converter.field_count, converter.feature_count))
#
## Model parameters
#LEARNING_RATE = 0.2
#LAMBDA = 0.002
#EPOCH = 10
#OPT_METHOD = "sgd" # options are "sgd", "adagrad" and "ftrl"
#
## The metrics for binary classification options are "acc", "prec", "f1" and "auc"
## for regression, options are "rmse", "mae", "mape"
#METRIC = "auc"
#
## Paths
#YAML_FILE_NAME = "xDeepFM.yaml"
#TRAIN_FILE_NAME = "cretio_tiny_train"
#VALID_FILE_NAME = "cretio_tiny_valid"
#TEST_FILE_NAME = "cretio_tiny_test"
#MODEL_FILE_NAME = "model.out"
#OUTPUT_FILE_NAME = "output.txt"
#
#data_path = "criteo"
#yaml_file = os.path.join(data_path, YAML_FILE_NAME)
#train_file = os.path.join(data_path, TRAIN_FILE_NAME)
#valid_file = os.path.join(data_path, VALID_FILE_NAME)
#test_file = os.path.join(data_path, TEST_FILE_NAME)
#model_file = os.path.join(data_path, MODEL_FILE_NAME)
#output_file = os.path.join(data_path, OUTPUT_FILE_NAME)
#
#assets_url = "https://recodatasets.z20.web.core.windows.net/deeprec/xdeepfmresources.zip"
#assets_file = maybe_download(assets_url, work_directory=data_path)
#unzip_file(assets_file, data_path)
#
## train FM
#fm_model = xl.create_fm()
#
#fm_model.setTrain(train_file)     # Set the path of training dataset
#fm_model.setValidate(valid_file)  # Set the path of validation dataset
#
#param = {"task":"binary", 
#         "lr": 0.2, 
#         "lambda": 0.02, 
#         "metric": 'auc',
#         "epoch": 10,
#         "opt": 'sgd'}
#
#fm_model.fit(param, "./model_output")
#
#
#
#
#
#
## Training task
#ffm_model = xl.create_ffm()        # Use field-aware factorization machine (ffm)
#ffm_model.setTrain(train_file)     # Set the path of training dataset
#ffm_model.setValidate(valid_file)  # Set the path of validation dataset
#
## Parameters:
##  0. task: binary classification
##  1. learning rate: 0.2
##  2. regular lambda: 0.002
##  3. evaluation metric: auc
##  4. number of epochs: 10
##  5. optimization method: sgd
#param = {"task":"binary", 
#         "lr": LEARNING_RATE, 
#         "lambda": LAMBDA, 
#         "metric": METRIC,
#         "epoch": EPOCH,
#         "opt": OPT_METHOD
#        }
#
## Start to train
## The trained model will be stored in model.out
#with Timer() as time_train:
#    ffm_model.fit(param, model_file)
#print(f"Training time: {time_train}")
#
## Prediction task
#ffm_model.setTest(test_file)  # Set the path of test dataset
#ffm_model.setSigmoid()        # Convert output to 0-1
#
## Start to predict
## The output result will be stored in output.txt
#with Timer() as time_predict:
#    ffm_model.predict(model_file, output_file)
#print(f"Prediction time: {time_predict}")
#
#with open(output_file) as f:
#    predictions = f.readlines()
#
#with open(test_file) as f:
#    truths = f.readlines()
#
#truths = np.array([float(truth.split(' ')[0]) for truth in truths])
#predictions = np.array([float(prediction.strip('')) for prediction in predictions])
#
#auc_score = roc_auc_score(truths, predictions)
#
## hyperparameter tuning
#
#param_dict = {
#    "lr": [0.0001, 0.001, 0.01],
#    "lambda": [0.001, 0.01, 0.1]
#}
#
#param_grid = generate_param_grid(param_dict)
#
#auc_scores = []
#
#with Timer() as time_tune:
#    for param in param_grid:
#        ffm_model = xl.create_ffm()       
#        ffm_model.setTrain(train_file)     
#        ffm_model.setValidate(valid_file)
#        ffm_model.fit(param, model_file)
#
#        ffm_model.setTest(test_file)  
#        ffm_model.setSigmoid()        
#        ffm_model.predict(model_file, output_file)
#
#        with open(output_file) as f:
#            predictions = f.readlines()
#
#        with open(test_file) as f:
#            truths = f.readlines()
#
#        truths = np.array([float(truth.split(' ')[0]) for truth in truths])
#        predictions = np.array([float(prediction.strip('')) for prediction in predictions])
#
#        auc_scores.append(roc_auc_score(truths, predictions))
#
#        
#auc_scores = [float('%.4f' % x) for x in auc_scores]
#auc_scores_array = np.reshape(auc_scores, (len(param_dict["lr"]), len(param_dict["lambda"]))) 
#
#auc_df = pd.DataFrame(
#    data=auc_scores_array, 
#    index=pd.Index(param_dict["lr"], name="LR"), 
#    columns=pd.Index(param_dict["lambda"], name="Lambda")
#)
#auc_df

