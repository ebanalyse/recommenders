def starfish(x):
    return x*2

def helper(session, tfidf, news):
    hist = session[1]
    hist = news[news[0].isin(hist)]
    hist = hist[[3,4]]
    hist = hist.fillna("")
    hist = hist[3] + ". " + hist[4]
    hist = hist.str.cat(sep=". ")
    out = tfidf.transform([hist])