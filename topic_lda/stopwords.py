"""
stopwords.py

TF-IDF 통계 기반으로 전체 문서에서 중요도가 낮은 단어를 불용어 후보로 자동 추출하는 유틸리티 모듈.

1. 텍스트 리스트를 TF-IDF 벡터화
2. 평균 TF-IDF 값이 임계값 이하인 단어를 불용어로 선정

주요 함수:
    - tfidf_analyze()
    - select_stopwords()
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

### TF-IDF 분석
def tfidf_analyze(texts):
    """
    입력 텍스트 리스트에 대해 TF-IDF 분석을 수행하고, 단어별 평균 TF-IDF 점수를 계산합니다.

    Args:
        texts (List[str]): 문서 텍스트 리스트 (각 문서는 문자열)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - feature_name: 추출된 전체 단어 목록
            - mean_tfidf: 각 단어의 평균 TF-IDF 점수

    Prints:
        - 전체 TF-IDF 점수의 평균, 분산, 분위수 통계

    Example:
        >>> fnames, scores = tfidf_analyze(["은행 금리 인상", "미국 금리 하락"])
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
    평균 TF-IDF 점수가 주어진 임계값 이하인 단어를 불용어 후보로 선정합니다.

    Args:
        feature_name (np.ndarray): 단어 목록
        mean_tfidf (np.ndarray): 각 단어의 평균 TF-IDF 점수
        threshold (float, optional): 불용어로 간주할 평균 TF-IDF 최대값. 기본값은 0.0007

    Returns:
        List[str]: 불용어 후보 단어 리스트

    Prints:
        - 선택된 불용어 후보 개수

    Example:
        >>> stopwords = select_stopwords(fnames, scores, threshold=0.001)
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