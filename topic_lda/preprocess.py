"""
preprocess.py

스크립트 텍스트 파일들을 불러와 전처리 및 토크나이징을 수행하는 모듈입니다.

주요 처리 흐름:
    1. 파일 인코딩 자동 감지
    2. 텍스트 파일 읽기 (두 번째 줄 기준)
    3. 숫자 및 공백만 있는 문장 제거
    4. 명사만 추출하여 토큰화된 문자열 리스트 생성

사용 예시:
    file_list = load_scripts('./download_folder')
    docs = read_scripts(file_list)
    clean_docs = remove_space_num(docs)
    texts = token_doc(clean_docs)
"""

import os
import chardet
from CustomTokenizer import CustomTokenizer
from tqdm import tqdm

### 파일 인코딩을 감지하는 함수
def detect_encoding(file_path):
    """
    텍스트 파일의 문자 인코딩을 자동으로 감지합니다.

    Args:
        file_path (str): 텍스트 파일 경로

    Returns:
        str: 감지된 인코딩 문자열 (예: 'utf-8', 'euc-kr')
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

### 스크립트 txt 파일 목록 읽어오기
def load_scripts(folder_path):
    """
    지정한 폴더 내 모든 텍스트(.txt) 파일 경로를 재귀적으로 수집합니다.

    Args:
        folder_path (str): 루트 폴더 경로

    Returns:
        List[str]: 전체 텍스트 파일 경로 리스트
    """
    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list

### txt 파일 내용 읽어오기
def read_scripts(file_list):
    """
    주어진 파일 리스트에서 두 번째 줄 텍스트만 읽어 문서 리스트를 만듭니다.

    Args:
        file_list (List[str]): 텍스트 파일 경로 리스트

    Returns:
        List[str]: 각 파일에서 추출한 한 줄 텍스트 리스트

    Notes:
        - 인코딩은 자동 감지
        - 읽기 실패한 파일은 경고 메시지만 출력
        - 두 번째 줄이 없으면 IndexError 가능 (try-except로 무시됨)
    """
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
    """
    공백이거나 숫자만 포함된 문장을 제거합니다.

    Args:
        documents (List[str]): 문장 리스트

    Returns:
        List[str]: 전처리된 문장 리스트
    """
    preprocessed_documents = []

    for line in tqdm(documents):
        if line and not line.replace(' ', '').isdecimal():
            preprocessed_documents.append(line)

    return preprocessed_documents

### 문서 내용에서 명사만 남기기
def token_doc(preprocessed_documents):
    """
    전처리된 문장에서 명사(NNG, NNP)만 추출하여 토큰 문자열로 변환합니다.

    Args:
        preprocessed_documents (List[str]): 전처리된 문장 리스트

    Returns:
        List[str]: 명사만 추출된 공백 구분 텍스트 리스트

    Example:
        Input 문장: "한국은행은 기준금리를 인상했다."
        Output: ["한국은행 기준 금리"]
    """
    texts = []
    tokenizer = CustomTokenizer()

    for news in tqdm(preprocessed_documents):
        texts.append(' '.join(tokenizer(news)))

    return texts