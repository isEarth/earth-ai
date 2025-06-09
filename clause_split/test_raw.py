import torch
from transformers import AutoTokenizer
from train import Config, LabelData, TaggingModel
from prediction import prediction, ClauseSpliting, Significant

def watch_prediction(test_sentence):
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