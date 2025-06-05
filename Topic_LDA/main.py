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