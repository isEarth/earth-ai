from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import csv

### 불용어 처리된 용어 사전 구축
def build_dictionary(texts, stopword_candidates):
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
    corpus = [dictionary.doc2bow(text.split()) for text in texts]
    print("#최종적으로 문서에 있는 단어 수: %d개"%len(dictionary))
    print("#카운트 벡터 수: %d개"%len(corpus))

    return corpus

### lda 모델링 및 저장
def lda_modeling(corpus, dictionary, num_topics=30, passes=10):
    model = LdaModel(corpus=corpus,id2word=dictionary,\
        passes=passes, num_topics=num_topics,\
            random_state=7)

    model.save("yt_topics")
    print("LDA 모델 저장됨: yt_topics")
    return model

### 토픽 목록 csv 저장
def save_topics(model):
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