"""
lda_topic_model.py

텍스트 리스트로부터 Gensim 기반 LDA 토픽 모델을 학습하고, 주요 토픽 단어를 CSV로 저장하는 전체 파이프라인을 포함합니다.

- 불용어 기반 단어 필터링
- LDA 모델 학습 및 저장
- 주요 토픽 추출 및 CSV 저장

사용 라이브러리: Gensim, CSV
"""

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import csv

### 불용어 처리된 용어 사전 구축
def build_dictionary(texts, stopword_candidates):
    """
    텍스트 리스트로부터 Gensim Dictionary 객체를 생성하고, 불용어 후보를 제거합니다.

    Args:
        texts (List[str]): 공백 기준 토큰화된 문장 리스트
        stopword_candidates (List[str]): 제거할 불용어 단어 리스트

    Returns:
        Dictionary: 불용어 제거된 Gensim Dictionary 객체
    """
    text_tokens = []

    for t in texts:
        text_tokens.append(t.split())

    # 리스트로 토큰화 후 dictionary 생성
    dictionary = Dictionary(text_tokens)
    print("#문서에 있는 단어 수: %d개"%len(dictionary))

    stopword_ids = [dictionary.token2id[word] for word in stopword_candidates if word in dictionary.token2id]
    dictionary.filter_tokens(stopword_ids)
    print("#IF-IDF 불용어 단어를 제외하고 문서에 남아 있는 단어 수: %d개"%len(dictionary))

    return dictionary

### 카운트 벡터 생성
def build_corpus(dictionary, texts):
    """
    Gensim Dictionary를 기반으로 문서별 BoW(Bag-of-Words) 벡터를 생성합니다.

    Args:
        dictionary (Dictionary): Gensim Dictionary 객체
        texts (List[str]): 공백 기준 토큰화된 문장 리스트

    Returns:
        List[List[Tuple[int, int]]]: 각 문서의 BoW 표현
    """
    corpus = [dictionary.doc2bow(text.split()) for text in texts]
    print("#최종적으로 문서에 있는 단어 수: %d개"%len(dictionary))
    print("#카운트 벡터 수: %d개"%len(corpus))

    return corpus

### lda 모델링 및 저장
def lda_modeling(corpus, dictionary, num_topics=30, passes=10):
    """
    LDA 토픽 모델을 학습하고 저장합니다.

    Args:
        corpus (List[List[Tuple[int, int]]]): BoW 표현 문서 리스트
        dictionary (Dictionary): Gensim Dictionary 객체
        num_topics (int): 추출할 토픽 개수 (기본값: 30)
        passes (int): 학습 반복 횟수 (기본값: 10)

    Returns:
        LdaModel: 학습된 Gensim LDA 모델 객체
    """
    model = LdaModel(corpus=corpus,id2word=dictionary,\
        passes=passes, num_topics=num_topics,\
            random_state=7)

    model.save("yt_topics")
    print("LDA 모델 저장됨: yt_topics")
    return model

### 토픽 목록 csv 저장
def save_topics(model):
    """
    LDA 모델에서 토픽별 상위 단어 5개를 추출하여 CSV로 저장합니다.

    Args:
        model (LdaModel): 학습된 Gensim LDA 모델 객체

    Output:
        - 파일명: topics.csv
        - 형식: Topic 번호, 단어1, 단어2, 단어3, 단어4, 단어5
    """
    data = model.print_topics(num_words=5, num_topics=50)

    with open('topics.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 헤더 작성
        writer.writerow(['Topic', 'Word1', 'Word2', 'Word3', 'Word4', 'Word5'])

        # 각 토픽 처리
        for topic_num, topic_string in data:
            # 단어와 가중치를 분리
            words = [word.split('*')[1].strip('"') for word in topic_string.split(' + ')]
            # 토픽 번호와 단어들을 CSV 행으로 작성
            writer.writerow([topic_num] + words)

    print("CSV 파일이 'topics.csv'로 저장되었습니다.")