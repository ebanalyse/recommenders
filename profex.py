import pandas as pd
from reco_utils.dataset.mind import download_mind, read_clickhistory
import os

mind_path = "mind/small"
news = pd.read_csv(f"{mind_path}/train/news.tsv", sep="\t", header=None)
sessions, history = read_clickhistory(os.path.join(mind_path, "train"), "behaviors.tsv")

def helper(session, news):
    hist = session[1]
    hist = news[news[0].isin(hist)]
    hist = hist[[3,4]]
    hist = hist.fillna("")
    hist = hist[3] + ". " + hist[4]
    hist = hist.str.cat(sep=". ")
    return hist


texts = list(map(lambda x: helper(x, news=news), sessions))
