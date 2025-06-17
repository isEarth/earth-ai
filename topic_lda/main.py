from preprocess import *
from stopwords import *
from lda_modeling import *

# ========================================
# 설정: 스크립트 텍스트가 저장된 디렉토리
# ========================================
FILE_PATH = './download_folder'

# ========================================
# STEP 1: 텍스트 전처리
# ========================================
print("전처리 중......")

file_list = load_scripts(FILE_PATH)                     # 파일 경로 수집
documents = read_scripts(file_list)                     # 파일 내용 읽기
preprocessed_documents = remove_space_num(documents)    # 숫자/공백 줄 제거
texts = token_doc(preprocessed_documents)               # 명사 토큰화 (띄어쓰기 연결)

# ========================================
# STEP 2: TF-IDF 기반 불용어 사전 구축
# ========================================
print("불용어 사전 구축 중......")

feature_name, mean_tfidf = tfidf_analyze(texts)         # 평균 TF-IDF 계산
stopword_candidates = select_stopwords(                 # 낮은 정보량 단어 추출
    feature_name,
    mean_tfidf,
    threshold=0.0007
)

# ========================================
# STEP 3: 카운트 벡터 생성 (BoW)
# ========================================
print("카운트 벡터 생성 중......")

dictionary = build_dictionary(texts, stopword_candidates)  # 불용어 제거된 Dictionary 생성
corpus = build_corpus(dictionary, texts)                   # 문서별 BoW 생성

# ========================================
# STEP 4: LDA 토픽 모델링
# ========================================
print("LDA 모델링 중......")

model = lda_modeling(                                      # LDA 학습
    corpus=corpus,
    dictionary=dictionary,
    num_topics=30,
    passes=10
)

save_topics(model)                                         # 토픽 → CSV 저장