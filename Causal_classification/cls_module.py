import argparse
import re
import os
from tqdm import tqdm
import kss
import pandas as pd
from transformers import pipeline, AutoTokenizer

# python cls_module.py -test golden_causal.csv
# python cls_module.py -test input_test.txt

MODEL_NAME = "kakaobank/kf-deberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

MODEL_PATH = '/home/eunhyea/EARTH/causal/runs/run_20250603_202042/best_model'
clf = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=tokenizer,
    device=0  # GPU 사용 시 주석 해제
)

def classify_sentences(sentences_with_labels):
    results = []
    for sentence, golden_label in tqdm(sentences_with_labels, desc="문장별 인과분류 진행"):
        output = clf(sentence)[0]
        label = output["label"]
        score = output["score"]
        results.append({
            "문장": sentence,
            "골든라벨": golden_label,
            "예측라벨": 1 if label == "LABEL_1" else 0,
            "예측": "포함" if label == "LABEL_1" else "미포함",
            "신뢰도": round(score, 4)
        })
    return pd.DataFrame(results)

def process_txt(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        text = re.sub('\n', ' ', text)
    sentences = [(s, None) for s in kss.split_sentences(text)]
    return classify_sentences(sentences)

def process_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"입력 CSV 파일이 존재하지 않습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'sentence' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV 파일에 'sentence' 및 'label' 컬럼이 필요합니다.")
    sentences_with_labels = list(zip(df['sentence'].tolist(), df['label'].tolist()))
    return classify_sentences(sentences_with_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="인과관계 분류 모듈")
    parser.add_argument(
        "-test",
        type=str,
        required=True,
        help="테스트 파일 이름 (예: golden.csv)."
    )
    args = parser.parse_args()

    stem, ext = os.path.splitext(args.test)
    ext = ext.lower()
    output_path = f"{MODEL_PATH}/{stem}_result.csv"

    if ext == ".csv":
        df = process_csv(args.test)
    elif ext == ".txt":
        df = process_txt(args.test)
    else:
        raise ValueError(f"지원하지 않는 확장자: {ext} (csv/txt 만 허용)")

    if df['골든라벨'].isna().sum():
        df.drop(columns='골든라벨', inplace=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(df)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(df)
    print(f"✅ 분류 결과가 저장되었습니다: {output_path}")
