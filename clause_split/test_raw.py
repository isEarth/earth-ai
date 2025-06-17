import torch
from transformers import AutoTokenizer
from train import Config, LabelData, TaggingModel
from prediction import prediction, ClauseSpliting, Significant

def watch_prediction(test_sentence):
    """
    문장을 BIO 태깅 모델로 예측하여, 각 토큰의 레이블과 신뢰도를 출력합니다.

    Args:
        test_sentence (str): 예측하고자 하는 단일 문장

    Prints:
        각 토큰에 대한 태깅 결과:
            - 태그 (예: E, O 등)
            - 신뢰도 (confidence)
            - 색상 강조 (인과 태그는 보라색, 신뢰도에 따라 파랑/빨강)
    """
    # setting
    config = Config()
    label_data = LabelData()
    model = TaggingModel(config)
    model.load_state_dict(torch.load("clause_model_earth.pt"))
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    # prediction
    results = prediction(model, tokenizer, test_sentence, label_data.id2label)

    # print
    for tok, label, confidence in results:
        if label in ['E','E2','E3']:
            color  = ('\033[95m','\033[0m')
            color2 = ('\033[94m','\033[0m') if confidence > 1.0 else ('\033[91m','\033[0m')
        else:
            color  = ('','')
            color2 = ('','')
        print(f"{color[0]}{tok}\t\t→ {label}{color[1]}\t {color2[0]}{confidence}{color2[1]}")


def watch_highlight(file_name):
    """
    텍스트 파일의 문장을 구문(clause) 단위로 분할하고,
    각 구문에 대해 중요 키워드를 하이라이팅한 결과를 출력합니다.

    Args:
        file_name (str): 문장 목록이 포함된 텍스트(.txt) 파일 경로

    Prints:
        - 구문 단위 분할 결과 (슬래시로 구분)
        - 각 구문별로 강조된 핵심 키워드
        - 전체 문장을 기준으로 추출된 강조 키워드
    """
    config = Config()
    config.confidence_threshold = 0.15

    with open(file_name, 'r', encoding='utf-8-sig') as f:
        raw = f.read()
        sentences = [r for r in raw.splitlines()]

    r = ClauseSpliting(sentences, e_option= 'E3', threshold= True)

    for rr in r.splited:
        print(' / '.join(rr))
        print(' / '.join([Significant(rrr,config).highlighted for rrr in rr]))
        print(Significant(' '.join(rr),config).highlighted)
        print()


test_sentence = "정부의 보조금 지급은 특정 산업의 단기 경쟁력을 높일 수 있으나, 장기적으로는 비효율을 초래할 수 있다."
watch_prediction(test_sentence)

watch_highlight('example2.txt')