import os
import chardet
from CustomTokenizer import CustomTokenizer
from tqdm import tqdm

### 파일 인코딩을 감지하는 함수
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

### 스크립트 txt 파일 목록 읽어오기
def load_scripts(folder_path):
    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list

### txt 파일 내용 읽어오기
def read_scripts(file_list):
    documents = []

    for idx in tqdm(range(len(file_list))):
        try:
            with open(file_list[idx], "r", encoding=detect_encoding(file_list[idx])) as file:
                text = file.read()
            documents.append(text.split('\n')[1])
        except:
            print(file_list[idx])

    return documents

### 빈 문자열이거나 숫자로만 이루어진 줄은 제외
def remove_space_num(documents):
    preprocessed_documents = []

    for line in tqdm(documents):
        if line and not line.replace(' ', '').isdecimal():
            preprocessed_documents.append(line)

    return preprocessed_documents

### 문서 내용에서 명사만 남기기
def token_doc(preprocessed_documents):
    texts = []
    tokenizer = CustomTokenizer()

    for news in tqdm(preprocessed_documents):
        texts.append(' '.join(tokenizer(news)))

    return texts