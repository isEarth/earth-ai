"""
causal_train.py

이 스크립트는 한국어 인과관계 문장 분류를 위한 DeBERTa 기반 텍스트 분류 모델을 학습합니다.

- 입력: CSV 파일 (./data/*.csv) - sentence, label
- 처리: 토큰화, 데이터 분할, 모델 학습, 평가지표 시각화
- 출력:
    - ./runs/run_YYYYMMDD_HHMMSS 디렉토리 내부에 모델 및 시각화 결과 저장

학습 대상:
    0: 비인과 문장
    1: 인과 문장

사용 모델:
    kakaobank/kf-deberta-base
"""

import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import torch

from tqdm import tqdm
from datetime import datetime

# ——————————————————————————————————————————————————————
def set_seed(seed):
    """
    재현성을 위한 랜덤 시드 고정 함수

    Args:
        seed (int): 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# plt 한글처리
plt.rcParams['font.family'] ='NanumGothic'
plt.rcParams['axes.unicode_minus'] =False

# ——————————————————————————————————————————————————————
# 1) 토크나이저와 분류 모델 로드

MODEL_NAME = "kakaobank/kf-deberta-base"
MODEL_PATH = '/home/eunhyea/EARTH/causal/causal_cls/checkpoint-'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # 0: 비인과, 1: 인과
)

# model     = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_PATH,
#     num_labels=2  # 0: 비인과, 1: 인과
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# ——————————————————————————————————————————————————————
def compute_metrics(pred):
    """
    Trainer 검증 단계에서 평가 지표 계산

    Args:
        pred (EvalPrediction): 예측 결과 객체

    Returns:
        dict: accuracy, precision, recall, f1, roc_auc
    """
    logits, labels = pred.predictions, pred.label_ids
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, logits[:, 1])
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc}

# ——————————————————————————————————————————————————————
def load_and_split_csv(directory, test_size=0.2, seed=42):
    """
    가장 최신 CSV 파일을 로드하고 train/val 데이터셋으로 분리

    Args:
        directory (str): CSV 파일이 있는 디렉토리 경로
        test_size (float): 검증 데이터 비율
        seed (int): 랜덤 시드

    Returns:
        tuple: (train_ds, val_ds) 형태의 Hugging Face Dataset 객체
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"❌ '{directory}' 디렉토리에 CSV 파일이 없습니다.")

    # 가장 마지막 수정된 파일 찾기
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"✅ CSV 파일 로드: {latest_csv}")

    df = pd.read_csv(latest_csv)
    ds = Dataset.from_pandas(df)
    split = ds.train_test_split(test_size=test_size, seed=seed)

    train_ds = split["train"]
    val_ds = split["test"]

    return train_ds, val_ds


# ——————————————————————————————————————————————————————
def tokenize_fn(batch):
    """
    Dataset.map()에 넘기는 토크나이저 함수

    Args:
        batch (dict): {"sentence": List[str]}

    Returns:
        dict: tokenized inputs (input_ids, attention_mask)
    """
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# ——————————————————————————————————————————————————————
def plot_metrics(metrics, save_dir):
    """
    학습 로그(metric) 시각화

    Args:
        metrics (dict): metric 로그 딕셔너리
        save_dir (str): 저장 디렉토리 경로
    """
    for metric in ["loss", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_roc_auc"]:
        if metric in metrics:
            plt.figure()
            plt.plot(metrics[metric], label=metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} over epochs')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{metric}.png'))
            plt.close()


# ——————————————————————————————————————————————————————
def plot_confusion(predictions, labels, save_dir):
    """
    혼동행렬 시각화 및 저장

    Args:
        predictions (List[int]): 예측 레이블
        labels (List[int]): 실제 레이블
        save_dir (str): 저장 디렉토리
    """
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['비인과', '인과'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


# ——————————————————————————————————————————————————————
def plot_roc_auc(labels, probs, save_dir):
    """
    ROC 곡선 시각화 및 저장

    Args:
        labels (List[int]): 실제 레이블
        probs (List[float]): 양성 클래스 확률
        save_dir (str): 저장 디렉토리
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()


# ——————————————————————————————————————————————————————
def main():
    """
    전체 학습 파이프라인 실행:
        1. CSV 로드 및 데이터셋 분리
        2. 토큰화
        3. Trainer 설정 및 학습
        4. 모델 저장 및 시각화 결과 저장
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"./runs/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 1) train.csv -> train, val split
    print("📂 train.csv 불러오는 중...")
    train_ds, val_ds = load_and_split_csv("./data", test_size=0.2, seed=42)

    # 2) 토크나이즈: datasets.map() 에 desc 인자를 주면 tqdm 바가 자동으로 출력됩니다.
    print("🔍 train 데이터 토크나이즈 중...")
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing train set"  # tqdm 바 설명
    )

    print("🔍 val 데이터 토크나이즈 중...")
    val_ds   = val_ds.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing val set"    # tqdm 바 설명
    )

    # 3) Trainer 호환을 위해 컬럼명 변경
    print("✏️ 컬럼 이름 변경 중 (label → labels)...")
    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")

    # 4) 불필요한 컬럼(예: pandas 인덱스) 제거
    #    Dataset.from_pandas() 실행 시 "__index_level_0__" 컬럼이 남을 수 있습니다.
    if "__index_level_0__" in train_ds.column_names:
        train_ds = train_ds.remove_columns(["__index_level_0__"])
    if "__index_level_0__" in val_ds.column_names:
        val_ds   = val_ds.remove_columns(["__index_level_0__"])

    # 5) TrainingArguments & Trainer 설정
    args = TrainingArguments(
        output_dir                   = save_dir,
        learning_rate                = 2e-5,
        per_device_train_batch_size  = 16,
        per_device_eval_batch_size   = 16,
        num_train_epochs             = 20,
        evaluation_strategy          = "epoch",
        save_strategy                = "epoch",
        load_best_model_at_end       = True,
        metric_for_best_model        = "f1",
        fp16                         = True,  # GPU 환경이 아니면 False로 바꿔주세요
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6) 학습 시작
    print("🚀 학습 시작...")
    trainer.train()

    # 7) 학습 완료 후, 최고 성능 모델 저장
    trainer.save_model(os.path.join(save_dir, "best_model"))
    print("✅ 학습 완료 및 모델 저장됨: ./causal_cls-best")

    # 8) 평가지표 저장
    metrics = trainer.state.log_history
    metrics_dict = {}
    for entry in metrics:
        for key, value in entry.items():
            if key not in metrics_dict:
                metrics_dict[key] = []
            metrics_dict[key].append(value)
    plot_metrics(metrics_dict, save_dir)

    outputs = trainer.predict(val_ds)
    preds = np.argmax(outputs.predictions, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(outputs.predictions), dim=-1)[:, 1].numpy()
    labels = outputs.label_ids

    plot_confusion(preds, labels, save_dir)
    plot_roc_auc(labels, probs, save_dir)
    print(f"✅ 모든 그래프와 결과가 {save_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()