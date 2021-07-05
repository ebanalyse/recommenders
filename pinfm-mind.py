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
from multiprocessing.pool import ThreadPool
import re
import time

# download data set
mind_path = "mind"
dataset = "demo"
download = False
add_news_features = True

if download:
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
        'user_id': [session[0]] * len(labels),
        'news_id': session[2] + session[3],
        'label': labels
        }

    data_out = pd.DataFrame.from_dict(data_out, orient="columns")

    return data_out

def get_data(dataset="train", mind_path="mind", n_processes=None):

    sessions, history = read_clickhistory(os.path.join(mind_path, dataset), "behaviors.tsv")

    start_time = time.time()

    #if n_processes is None:
    #    n_processes = mp.cpu_count() * 2

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

if add_news_features:
    news = pd.read_csv(f"{mind_path}/train/news.tsv", sep="\t", header=None)
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int)
    enc.fit(news[1].values.reshape(-1, 1))

# one-hot encode news category
def add_news_features(df, data, mind_path, enc):

    #import pdb; pdb.set_trace()
    news = pd.read_csv(f"{mind_path}/{data}/news.tsv", sep="\t", header=None)
    news = news.rename(columns={0: 'news_id', 1: 'cat_txt'})

    news = pd.merge(df["news_id"], news[["news_id", "cat_txt"]], how="left", on="news_id")
    topic = pd.DataFrame(enc.transform(news["cat_txt"].values.reshape(-1, 1)))
    topic = topic.add_prefix("cat_")

    # add to training data
    trn = df.reset_index(drop=True)
    topic = topic.reset_index(drop=True)
    df = pd.concat([trn, topic], axis = 1)

    return df

if add_news_features:
    trn = add_news_features(df=trn,
                            data="train",
                            mind_path=mind_path,
                            enc=enc)

converter = LibffmConverter().fit(trn, col_rating='label')

def transform_export(dataset, converter, file, mind_path="mind"):

    start_time = time.time()

    print(f"Converting {len(dataset)} observations to libffm format")

    out = converter.transform(dataset)

    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))    

    # export data
    out_path = os.path.join(mind_path, file)
    out.to_csv(out_path, sep=' ', index=False, header=False)

    return out_path

# transform and export train and valid
data = "train"
# trn = get_data(data, mind_path=mind_path)
# trn = add_news_features(df=trn,
#                         data=data,
#                         mind_path=mind_path,
#                         enc=enc)
file_train = f"{data}.txt"
transform_export(trn, converter, file_train, mind_path=mind_path)

data = "dev" 
file_dev = f"{data}.txt"
dev = get_data("dev",mind_path=mind_path)
if add_news_features:
    dev = add_news_features(df=dev,
                            data=data,
                            mind_path=mind_path,
                            enc=enc)
    transform_export(dev, converter, file_dev, mind_path=mind_path)

# train FM
fm_model = xl.create_fm()

fm_model.setTrain(os.path.join(mind_path, file_train))     # Set the path of training dataset
fm_model.setValidate(os.path.join(mind_path, file_dev))  # Set the path of validation dataset

param = {"task":"binary", 
         "lr": 0.3, 
         "lambda": 0.02, 
         "metric": 'auc',
         "stop_window": 5,
         "epoch": 10,
         "opt": 'sgd'}

fm_model.fit(param, "./model_output")

#TF-IDF
news = pd.read_csv(f"{mind_path}/train/news.tsv", sep="\t", header=None)
corpus = news[[3,4]]
corpus = corpus.fillna("")
corpus = corpus[3] + ". " + corpus[4]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10)
X = tfidf.fit(corpus)
tfidf.transform(corpus)

sessions, history = read_clickhistory(os.path.join(mind_path, "train"), "behaviors.tsv")
hist = sessions[0][1]

hist = news[news[0].isin(hist)]
hist = hist[[3,4]]
hist = hist.fillna("")
hist = hist[3] + ". " + hist[4]
tfidf.transform(hist)

#news[0] in ["N3112"]
#(news[0] in hist).any()













