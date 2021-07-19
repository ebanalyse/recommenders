import sys
import os
from tempfile import TemporaryDirectory
import xlearn as xl
from xlearn.data import DMatrix
from sklearn.metrics import roc_auc_score
import numpy as np
#import os
#os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Rayimport pandas as pd
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from reco_utils.common.constants import SEED
from reco_utils.common.timer import Timer
from reco_utils.dataset.download_utils import maybe_download, unzip_file
from reco_utils.tuning.parameter_sweep import generate_param_grid
from reco_utils.dataset.mind import download_mind, read_clickhistory
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from itertools import compress
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import re
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datatable import dt, f, by, g, join, sort, update, ifelse, rowany
import numpy as np
from multiprocessing import Pool
from parallel import apply_tfidf
import math
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import scipy
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score

# CONFIG
mind_path = "mind"
dataset = "demo"
download = False
add_news_features = False
tfidf_k = 3000

#import nltk
#nltk.download('stopwords')

if download:
    download_mind(dataset, mind_path)
    unzip_file(os.path.join(mind_path, f"MIND{dataset}_train.zip"), dst_dir=os.path.join(mind_path, dataset, "train"))
    unzip_file(os.path.join(mind_path, f"MIND{dataset}_dev.zip"), dst_dir=os.path.join(mind_path, dataset, "dev"))

mind_path = os.path.join(mind_path, dataset)

def create_useritem(session):

    # create 'labels' vector
    labels = [1] * len(session[2])
    labels.extend([0] * len(session[3]))

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

    # print(f"processing {len(sessions)} impression logs with {n_processes} processes...")

    # TODO: prøv med process pool
    #pool = ThreadPool(processes=n_processes)
    #out = pool.map(create_useritem, sessions)
    out = [create_useritem(session) for session in sessions]

    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))    
    
    out = pd.concat(out)

    return out

#if add_news_features:
#    news = pd.read_csv(f"{mind_path}/train/news.tsv", sep="\t", header=None)
#    from sklearn.preprocessing import OneHotEncoder
#    enc = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int)
#    enc.fit(news[1].values.reshape(-1, 1))

# one-hot encode news category
#def add_news_features(df, data, mind_path, enc):
#
#    #import pdb; pdb.set_trace()
#    news = pd.read_csv(f"{mind_path}/{data}/news.tsv", sep="\t", header=None)
#    news = news.rename(columns={0: 'news_id', 1: 'cat_txt'})
#
#    news = pd.merge(df["news_id"], news[["news_id", "cat_txt"]], how="left", on="news_id")
#    topic = pd.DataFrame(enc.transform(news["cat_txt"].values.reshape(-1, 1)))
#    topic = topic.add_prefix("cat_")
#
#    # add to training data
#    trn = df.reset_index(drop=True)
#    topic = topic.reset_index(drop=True)
#    df = pd.concat([trn, topic], axis = 1)
#
#    return df
#
#if add_news_features:
#    trn = add_news_features(df=trn,
#                            data="train",
#                            mind_path=mind_path,
#                            enc=enc)

#TF-IDF
# Fit TF-IDF
news = pd.read_csv(f"{mind_path}/train/news.tsv", sep="\t", header=None)
corpus = news[[3,4]]
corpus = corpus.fillna("")
corpus = corpus[3] + ". " + corpus[4]
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = stopwords.words('english')
tfidf = TfidfVectorizer(stop_words = stop_words, max_features=tfidf_k, dtype=np.float32)
tfidf.fit(corpus)

def helper_sort(x):
    return x[0]

def apply_tfidf_parallel_hist(df, tfidf, mind_path=mind_path, part="train", n_processes=None):
    
    if n_processes is None:
        n_processes = mp.cpu_count()

    # load data
    news = pd.read_csv(f"{mind_path}/{part}/news.tsv", sep="\t", header=None)
    sessions, history = read_clickhistory(os.path.join(mind_path, part), "behaviors.tsv")

    # sort values in order to concatenate data later
    df = df.sort_values("user_id") 
    sessions.sort(key=helper_sort)

    # subset histories for unique users (since history is fixed(!))
    userids = []
    bl = []
    for session in sessions:
        userid = session[0]
        if userid in userids:
            bl.append(False)
        else:
            bl.append(True)
            userids.append(userid)

    sessions = compress(sessions, bl)

    # extract click histories
    sessions = [session[1] for session in sessions]

    # split data into equal-sized chunks
    chunks = np.array_split(sessions, 12)
    chunks = [{'sessions':x, 'news':news, 'tfidf':tfidf} for x in chunks] 

    # apply tdidf transform in parallel
    print(f"Collecting texts from {part} click history - {len(sessions)} sessions")
    start_time = time.time()
    p=Pool(n_processes)
    out = p.map(apply_tfidf, chunks)
    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))    

    idx = []
    val = ""
    i = -1
    uids = df["user_id"].values
    for id in uids:
        if id != val:
            i += 1
            val = id
        idx.append(i)   

    out = scipy.sparse.vstack(out).tocsr()
    n_chunks = math.ceil(len(idx) / 1e06)
    chunks = np.array_split(idx, n_chunks)
    chunks = [x.tolist() for x in chunks]
    def subset(x):
        return out[x]
    out = list(map(subset, chunks))
    out = scipy.sparse.vstack(out).tocsr()
    out = pd.DataFrame.sparse.from_spmatrix(out)
    out = out.add_prefix("tfidf_hist")
    out = out.reset_index(drop=True)
    df = df.reset_index(drop=True)
    out = pd.concat([df, out], axis=1)
    return out

# fit data preprocessing
trn = get_data("train", mind_path=mind_path)

start_time = time.time()
print("Apply TFIDF to train....")
trn = apply_tfidf_parallel_hist(df=trn, mind_path=mind_path, part="train", tfidf=tfidf)
seconds = time.time() - start_time
print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))  

# open a file, where you ant to store the data
#file = open('train.pkl', 'wb')

# dump information to that file
#pickle.dump(trn, file)

# close the file
# file.close()


categorical_features = ['user_id', 'news_id']
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=True, dtype=np.int)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)]) 

preprocessor.fit(trn)

#preprocessor.transform(trn)
def to_Xy(x):

    y = x["label"].values

    user_item = preprocessor.transform(x)

    X = x.drop(columns=['news_id', 'user_id', 'label'])
    X = scipy.sparse.csr_matrix(X.values)
    
    X = scipy.sparse.hstack([user_item, X]).tocsr()

    return X, y

X_trn, y_trn = to_Xy(trn)

#converter = LibffmConverter().fit(trn, col_rating='label')

#def transform_export(dataset, converter, file, mind_path="mind", n_processes=None):
#
#    start_time = time.time()
#
#    print(f"Converting {len(dataset)} observations to libffm format")
#
#    #if n_processes is None:
#    #    n_processes = mp.cpu_count()
#
#    # split data frame in chunks
#    #n_chunk = math.ceil(len(dataset) / n_processes)
#    #dataset = np.array_split(dataset, n_chunk)
#
#    #p=Pool(n_processes)
#    # out = p.map(converter.transform, dataset)
#    out = converter.transform(dataset)
#    #out = pd.concat(out, axis=1)
#
#    seconds = time.time() - start_time
#    print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))    
#
#    # export data
#    out_path = os.path.join(mind_path, file)
#    out.to_csv(out_path, sep=' ', index=False, header=False)
#
#    return out_path

# transform and export train and valid
#data = "train"
#trn = get_data(data, mind_path=mind_path)
#trn = apply_tfidf_parallel(df=trn, mind_path=mind_path, part=data, tfidf=tfidf)
# trn = add_news_features(df=trn,
#                         data=data,
#                         mind_path=mind_path,
#                         enc=enc)
#file_train = f"{data}.txt"
#transform_export(trn, converter, file_train, mind_path=mind_path)

data = "dev" 
#file_dev = f"{data}.txt"
dev = get_data("dev",mind_path=mind_path)
dev = apply_tfidf_parallel_hist(df=dev, mind_path=mind_path, part=data, tfidf=tfidf)
X_dev, y_dev = to_Xy(dev)

# open a file, where you ant to store the data
#file = open('dev.pkl', 'wb')

# dump information to that file
#pickle.dump(trn, file)

# close the file
# file.close()

from pyfm import pylibfm

fm = pylibfm.FM(num_factors=10, 
                num_iter=2, 
                verbose=True, 
                task="classification", 
                initial_learning_rate=0.001, 
                learning_rate_schedule="optimal")

start_time = time.time()
print("Training model....")
fm.fit(X_trn, y_trn)   
seconds = time.time() - start_time
print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))  

preds = fm.predict(X_dev)
roc_auc_score(y_dev, preds)

# xlearn
#if add_news_features:
#    dev = add_news_features(df=dev,
#                            data=data,
#                            mind_path=mind_path,
#                            enc=enc)
#transform_export(dev, converter, file_dev, mind_path=mind_path)

## train FM
#fm_model = xl.create_fm()
#
#fm_model.setTrain(os.path.join(mind_path, file_train))     # Set the path of training dataset
#fm_model.setValidate(os.path.join(mind_path, file_dev))  # Set the path of validation dataset
#
#param = {"task":"binary", 
#         "lr": 0.3, 
#         "lambda": 0.02, 
#         "metric": 'auc',
#         "stop_window": 5,
#         "epoch": 10,
#         "opt": 'sgd'}


        
        
    





