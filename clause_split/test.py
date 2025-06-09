import torch
from transformers import AutoTokenizer
from train import Config, LabelData, TaggingModel
from prediction import ClauseSpliting, prediction, highlight_jsonl, FileNames

def watch_prediction(test_sentence):
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
