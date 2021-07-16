from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
# from tester import helper
import numpy as np

# corpus = ["Ko", "Fisk"]
# tfidf = TfidfVectorizer(max_features=1)
# tfidf.fit(corpus)
# 
# mange = corpus*100
# 
# tfidf.transform(mange)
# 
# chunks = np.array_split(np.array(mange), 12)
# chunks = [(x, tfidf, i) for i, x in enumerate(chunks)]
# 
# p=Pool(12)
# 
# out = p.starmap(helper, chunks)

def get_history_texts(session, news):
    hist = news[news[0].isin(session)]
    hist = hist[[3,4]]
    hist = hist.fillna("")
    hist = hist[3] + ". " + hist[4]
    hist = hist.str.cat(sep=". ")
    return hist

def apply_tfidf(pkg):
    news = pkg.get('news')
    out = [get_history_texts(session, news) for session in pkg.get('sessions')]
    out = pkg.get('tfidf').transform(out)
    return out

