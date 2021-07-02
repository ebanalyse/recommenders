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
from reco_utils.dataset.mind import download_mind, read_clickhistory
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from itertools import compress
import multiprocessing as mp
from multiprocessing.pool import
import re
import time

# download data set
mind_path = "mind"
dataset = "small"
download_mind(dataset, mind_path)

unzip_file(os.path.join(mind_path, f"MIND{dataset}_train.zip"), dst_dir=os.path.join(mind_path, dataset, "train"))
unzip_file(os.path.join(mind_path, f"MIND{dataset}_dev.zip"), dst_dir=os.path.join(mind_path, dataset, "dev"))

mind_path = os.path.join(mind_path, dataset)

def create_useritem(session):

    # create 'labels' vector
    labels = [1] * len(session[2])
    labels.extend([0] * len(session[3]))

    len(labels)

    data_out = {
        'userid': [session[0]] * len(labels),
        'newsid': session[2] + session[3],
        'label': labels
        }

    data_out = pd.DataFrame.from_dict(data_out, orient="columns")

    return data_out

def get_data(dataset="train", mind_path="mind", n_processes=None):

    sessions, history = read_clickhistory(os.path.join(mind_path, dataset), "behaviors.tsv")

    start_time = time.time()

    if n_processes is None:
        n_processes = mp.cpu_count() * 2

    print(f"processing {len(sessions)} impression logs with {n_processes} processes...")

    # TODO: prøv med process pool
    #pool = ThreadPool(processes=n_processes)
    #out = pool.map(create_useritem, sessions)
    out = [create_useritem(session) for session in sessions]

    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))    
    
    out = pd.concat(out)

    return out

# fit data preprocessing
trn = get_data("train", mind_path=mind_path)
converter = LibffmConverter().fit(trn, col_rating='label')

def transform_export(dataset, converter, file, mind_path="mind"):

    # transform to libffm format
    out = converter.transform(dataset)

    # export data
    out_path = os.path.join(mind_path, file)
    out.to_csv(out_path, sep=' ', index=False, header=False)

    return out_path

# transform and export train and valid
file_train = "train.txt"
transform_export(trn, converter, file_train, mind_path=mind_path)

file_dev = "dev.txt"
dev = get_data("dev",mind_path=mind_path)
transform_export(dev, converter, file_dev, mind_path=mind_path)

# train FM
fm_model = xl.create_fm()

fm_model.setTrain(os.path.join(mind_path, file_train))     # Set the path of training dataset
fm_model.setValidate(os.path.join(mind_path, file_dev))  # Set the path of validation dataset

param = {"task":"binary", 
         "lr": 0.005, 
         "lambda": 0.02, 
         "metric": 'auc',
         "stop_window": 6,
         # "epoch": 50,
         "opt": 'sgd'}

fm_model.fit(param, "./model_output")
