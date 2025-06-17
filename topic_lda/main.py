"""
main.py

YouTube 스크립트 텍스트 파일로부터 LDA 기반 토픽 모델링을 수행하는 전체 파이프라인 스크립트.

1. 전처리:
    - 파일 불러오기 → 줄 단위 병합 → 공백/숫자 제거 → 명사 추출
2. TF-IDF 분석:
    - 중요도가 낮은 단어를 불용어 후보로 선정
3. 사전 및 코퍼스 생성:
    - 불용어 제거 후 Dictionary 및 BoW 벡터 생성
4. LDA 모델 학습:
    - Gensim 기반 모델 학습 및 저장
5. 결과 저장:
    - 토픽별 주요 단어 5개씩 CSV로 저장

의존 모듈:
    - preprocess.py: 텍스트 로딩 및 정제
    - stopwords.py: TF-IDF 기반 불용어 추출
    - lda_modeling.py: Dictionary, LDA 모델, CSV 저장
"""

from preprocess import *
from stopwords import *
from lda_modeling import *

### 스크립트가 한줄로 저장된 txt 파일들
FILE_PATH = './download_folder'

### 전처리
print("전처리 중......")
file_list = load_scripts(FILE_PATH)
documents = read_scripts(file_list)
preprocessed_documents = remove_space_num(documents)
texts = token_doc(preprocessed_documents)

### 불용어 사전 구축
print("불용어 사전 구축 중......")
feature_name, mean_tfidf = tfidf_analyze(texts)
stopword_candidates = select_stopwords(feature_name, mean_tfidf, threshold=0.0007)

### 카운트 벡터 생성
print("카운트 벡터 생성 중......")
dictionary = build_dictionary(texts, stopword_candidates)
corpus = build_corpus(dictionary, texts)

### lda 모델링
print("lda 모델링 중......")
model = lda_modeling(corpus, dictionary, num_topics=30, passes=10)
save_topics(model)