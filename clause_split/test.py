import torch
from transformers import AutoTokenizer
from train import Config, LabelData, TaggingModel
from prediction import ClauseSpliting, prediction, highlight_jsonl, FileNames

def watch_prediction(test_sentence):
    """
    문장을 BIO 태깅 기반 구문 분석 모델로 예측하여 각 토큰의 레이블(E, E2, E3 등)과 신뢰도를 시각적으로 출력합니다.

    Args:
        test_sentence (str): 예측하고자 하는 입력 문장

    Prints:
        토큰별 예측 결과를 색상 구분하여 콘솔에 출력합니다.
        - 레이블이 인과구문(E, E2, E3)인 경우 토큰을 보라색으로 강조
        - 신뢰도가 1.0 초과인 경우 파란색, 이하인 경우 빨간색으로 신뢰도 표시
    """
    # 모델 및 토크나이저 설정
    config = Config()
    label_data = LabelData()
    model = TaggingModel(config)
    model.load_state_dict(torch.load("clause_model_earth.pt"))
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    # 예측 수행
    results = prediction(model, tokenizer, test_sentence, label_data.id2label)

    # 결과 출력 (색상 포함)
    for tok, label, confidence in results:
        if label in ['E', 'E2', 'E3']:
            color  = ('\033[95m','\033[0m')  # 토큰 강조
            color2 = ('\033[94m','\033[0m') if confidence > 1.0 else ('\033[91m','\033[0m')  # 신뢰도 강조
        else:
            color = color2 = ('', '')
        print(f"{color[0]}{tok}\t→ {label}{color[1]}\t {color2[0]}{confidence}{color2[1]}")

def watch_highlight(file_name):
    """
    텍스트 파일 내 문장들을 절(clause) 단위로 분리하고, 각 절에서 의미 있는 단어를 강조하여 콘솔에 출력합니다.

    Args:
        file_name (str): 문장들이 줄 단위로 저장된 입력 텍스트 파일 경로

    Prints:
        각 문장을 절로 분할한 뒤, JSONL로 저장된 중요 단어 정보를 기반으로 강조 표시된 문장을 출력합니다.
        색상은 없지만 하이라이팅 구조가 적용된 형태로 출력됩니다.
    """
    config = Config()
    config.confidence_threshold = 0.15
    filenames = FileNames()

    # 텍스트 로드
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # 절 분리 수행
    r = ClauseSpliting(sentences, e_option='E3', threshold=True)

    # 강조 출력
    print(highlight_jsonl(filenames.significant_json))

# 테스트 예시 실행
test_sentence = "정부의 보조금 지급은 특정 산업의 단기 경쟁력을 높일 수 있으나, 장기적으로는 비효율을 초래할 수 있다."
watch_prediction(test_sentence)
watch_highlight('example2.txt')
