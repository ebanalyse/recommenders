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

# Download MovieLens
data_path = "movielens"

assets_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
assets_file = maybe_download(assets_url, work_directory=data_path)
unzip_file(assets_file, data_path)

data_path = os.path.join(data_path, "ml-latest-small")

# load data set
links = pd.read_csv(os.path.join(data_path, 'links.csv'))
movies = pd.read_csv(os.path.join(data_path, 'movies.csv'))
ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
tags = pd.read_csv(os.path.join(data_path, 'tags.csv'))


# compute and add content genre feature (just extract the first genre)
movies['genre'] = movies['genres'].apply(lambda x: x.split("|")[0])
df_genre = movies[["movieId", "genre"]]

ratings = ratings.merge(df_genre, how = "left", on="movieId")

# adjust col types (cathegorical)
ratings['userId'] = ratings['userId'].apply(str)
ratings['movieId'] = ratings['movieId'].apply(str)

# convert to (libsvm)libffm format
converter = LibffmConverter().fit(ratings, col_rating='rating')
out = converter.transform(ratings)

path_trn = "movielens/movielens_trn"
# out[['rating', 'userId', 'movieId']]
out.to_csv(path_trn, sep=' ', index=False, header=False)

# train FM
fm_model = xl.create_fm()

fm_model.setTrain(path_trn)     # Set the path of training dataset
fm_model.setValidate(path_trn)  # Set the path of validation dataset

param = {"task":"reg", 
         "lr": 0.2, 
         "lambda": 0.002}

fm_model.fit(param, "./model_output")

fm_model = xl.create_fm()
# dataset = "./testffm.txt"
dataset = "movielens/movielens_trn"
fm_model.setTrain(dataset)
fm_model.setValidate(dataset)
param = {'task':'reg', 'lr':0.2, 'lambda':0.002}

fm_model.fit(param, "./model.out")



