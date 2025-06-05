import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

### TF-IDF 분석
def tfidf_analyze(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_name = vectorizer.get_feature_names_out()
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()

    print("=========================================")
    print("평균:", mean_tfidf.mean())
    print("분산:", mean_tfidf.var())
    print("상위 10% 분위수:", np.percentile(mean_tfidf, 90))
    print("하위 10% 분위수:", np.percentile(mean_tfidf, 10))
    print("=========================================")

    return feature_name, mean_tfidf

### TF-IDF 기반 불용어 사전 구축
def select_stopwords(feature_name, mean_tfidf, threshold=0.0007):
    stopword_candidates = [
        feature_name[i]
        for i, avg in enumerate(mean_tfidf)
        if avg <= threshold
    ]

    print("=========================================")
    print("불용어 후보 개수:", len(stopword_candidates))
    print("=========================================")

    return stopword_candidates