import os
import chardet
from CustomTokenizer import CustomTokenizer
from tqdm import tqdm

### 파일 인코딩을 감지하는 함수
def detect_encoding(file_path):
    """
    주어진 파일의 문자 인코딩을 감지합니다.

    Args:
        file_path (str): 텍스트 파일 경로

    Returns:
        str: 감지된 인코딩 이름 (예: 'utf-8', 'euc-kr')
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

### 스크립트 txt 파일 목록 읽어오기
def load_scripts(folder_path):
    """
    지정된 폴더 하위에 있는 모든 파일의 경로를 수집합니다.

    Args:
        folder_path (str): 디렉토리 경로

    Returns:
        List[str]: 전체 파일 경로 리스트
    """
    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list

### txt 파일 내용 읽어오기
def read_scripts(file_list):
    """
    파일 리스트로부터 텍스트 내용을 읽어와 두 번째 줄만 추출합니다.

    Args:
        file_list (List[str]): 텍스트 파일 경로 리스트

    Returns:
        List[str]: 각 파일에서 추출된 문장 (두 번째 줄 기준)
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
    빈 문자열이거나 숫자로만 이루어진 문장을 필터링합니다.

    Args:
        documents (List[str]): 문장 리스트

    Returns:
        List[str]: 유효한 문장만 남긴 리스트
    """
    preprocessed_documents = []

    for line in tqdm(documents):
        if line and not line.replace(' ', '').isdecimal():
            preprocessed_documents.append(line)

    return preprocessed_documents

### 문서 내용에서 명사만 남기기
def token_doc(preprocessed_documents):
    """
    문장 리스트에서 명사(NNG, NNP)만 추출하여 반환합니다.

    Args:
        preprocessed_documents (List[str]): 전처리된 문장 리스트

    Returns:
        List[str]: 각 문장을 명사만 추출해 공백으로 연결한 텍스트 리스트
    """
    texts = []
    tokenizer = CustomTokenizer()

    for news in tqdm(preprocessed_documents):
        texts.append(' '.join(tokenizer(news)))

    return texts