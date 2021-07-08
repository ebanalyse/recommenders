def starfish(x):
    return x*2

def helper(x, tfidf, idx):
    out = tfidf.transform(x)
    return idx, out

def helpo(session, news):
    hist = news[news[0].isin(session)]
    hist = hist[[3,4]]
    hist = hist.fillna("")
    hist = hist[3] + ". " + hist[4]
    hist = hist.str.cat(sep=". ")
    return hist

def xhelper(pkg):
    news = pkg.get('news')
    out = [helpo(session, news) for session in pkg.get('sessions')]
    return out
