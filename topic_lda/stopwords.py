import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

### TF-IDF 분석
def tfidf_analyze(texts):
    """
    텍스트 집합에 대해 TF-IDF를 계산하고, 각 단어의 평균 TF-IDF 점수를 분석합니다.

    Args:
        texts (List[str]): 문서 리스트. 각 원소는 공백으로 구분된 토큰 문자열

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - feature_name: 단어 목록 (단어 인덱스 순)
            - mean_tfidf: 각 단어의 평균 TF-IDF 점수
    """
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
    """
    TF-IDF 평균 점수를 기반으로 정보량이 낮은 단어(불용어 후보)를 추출합니다.

    Args:
        feature_name (np.ndarray): 단어 목록
        mean_tfidf (np.ndarray): 각 단어의 평균 TF-IDF 점수
        threshold (float): 필터링 기준. 평균 TF-IDF가 이 값 이하이면 불용어로 간주

    Returns:
        List[str]: 불용어 후보 단어 리스트
    """
    stopword_candidates = [
        feature_name[i]
        for i, avg in enumerate(mean_tfidf)
        if avg <= threshold
    ]

    print("=========================================")
    print("불용어 후보 개수:", len(stopword_candidates))
    print("=========================================")

    return stopword_candidates