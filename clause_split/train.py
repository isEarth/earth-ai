# ===========================================================
# Token Classification 학습 파이프라인 (DeBERTa 기반)
# - 문장을 토큰 단위로 분류 (BIO 또는 절 경계 태그 등)
# - HuggingFace Transformers + Accelerate + PyTorch 기반
# ===========================================================
import os
import json
import random
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DebertaV2ForTokenClassification, AutoTokenizer, DebertaV2Model
from accelerate import Accelerator

# ——————————————————————————————————————————————————————
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# plt 한글처리
plt.rcParams['font.family'] ='NanumGothic'
plt.rcParams['axes.unicode_minus'] =False

# ------------------------------
# Config 클래스: 학습 설정 저장
# ------------------------------
@dataclass
class Config:
    model: str = "kakaobank/kf-deberta-base"
    dropout: float = 0.5
    max_length: int = 128
    batch_size: int = 1
    epochs: int = 100
    lr: float = 3e-4
    enable_scheduler: bool = True
    scheduler: str = 'CosineAnnealingWarmRestarts'
    gradient_accumulation_steps: int = 2
    adam_eps: float = 1e-6
    freeze_encoder: bool = True
    tag_weight: list = field(default_factory=lambda: [0.1, 1.0, 1.2, 1.2])  # 클래스 불균형 보정
    confidence_threshold: float = 0.5

# 라벨 매핑 클래스
@dataclass
class LabelData:
    labels: list = field(default_factory=lambda: ["O", "E", "E2", "E3"])
    id2label: dict = field(init=False)
    label2id: dict = field(init=False)

    def __post_init__(self):
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

# 글로벌 변수 저장용
@dataclass
class Variables:
    confidence_avg : float = 1.0

# ------------------------------
# WordPiece 토큰 복구 함수
# ------------------------------
def recover_wordpieces(tokens: list) -> str:
    words = []
    current_word = ''
    for token in tokens:
        if token.startswith('##'):
            current_word += token[2:]
        else:
            if current_word:
                words.append(current_word)
            current_word = token
    if current_word:
        words.append(current_word)
    try:
        if words[-1] == '.':
            words[-2] += '.'
            words.pop(-1)
    except:
        pass
    return ' '.join(words)

# ------------------------------
# BIO 태깅 텍스트 파일 로드 함수
# ------------------------------
def open_file(file_name):
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        raw = f.read()
        result, sents, tags = [], [], []
        for r in raw.splitlines():
            r = r.strip()
            if len(r) > 0:
                rr = r.split()
                if len(rr) != 2:
                    print("잘못된 줄 형식 감지됨")
                sents.append(rr[0])
                tags.append(rr[1])
            else:
                result.append({'tokens': sents, 'labels': tags})
                sents, tags = [], []

    print("The number of data sentences:", len(result))

    for r in result:
        r['full_text'] = recover_wordpieces(r["tokens"])

    return pd.DataFrame(result[:166])  # 앞 166개만 사용

# ------------------------------
# 토큰 분류용 커스텀 Dataset 클래스
# ------------------------------
class TokenTaggingDataset:
    def __init__(self, df, config, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.config = config
        self.label2id = LabelData().label2id

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['full_text']
        tokens = row['tokens']
        labels = row['labels']

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors='pt'
        )
        iter_labels = iter(labels)
        label_ids = []
        for id in encoding['input_ids'].squeeze():
            if id < 6 or id >= 130000:  # 특수토큰 영역 제외 (kf-deberta)
                label_ids.append(-100)
            else:
                try:
                    label = next(iter_labels)
                    label_id = self.label2id[label]
                    label_ids.append(label_id)
                except StopIteration:
                    label_ids.append(-100)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return len(self.df)

# ------------------------------
# Mean Pooling 클래스 (사용 안 됨)
# ------------------------------
class MeanPooling(nn.Module):
    def forward(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embed = torch.sum(hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return sum_embed / sum_mask

# ------------------------------
# 모델 클래스 정의
# ------------------------------
class TaggingModel(nn.Module):
    def __init__(self, config, num_classes=4):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(config.model, output_hidden_states=True)
        if config.freeze_encoder:
            for p in self.encoder.base_model.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, inputs, return_cls=False, out_last_hidden_state=False):
        out = self.encoder(**inputs, return_dict=True)
        sequence_output = self.dropout(out.last_hidden_state)
        logits = self.classifier(sequence_output)
        result = [logits]
        if return_cls:
            cls_vector = sequence_output[:, 0, :]
            result.append(cls_vector)
        if out_last_hidden_state:
            result.append(out.last_hidden_state)
        return result if any((return_cls, out_last_hidden_state)) else logits

# ------------------------------
# Trainer 클래스 (학습/검증)
# ------------------------------
class Trainer:
    def __init__(self, model, loaders, config, accelerator):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.confidence_avg = Variables().confidence_avg
        self.accelerator = accelerator
        self.optimizer = self._get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, eta_min=1e-7)
        self.train_losses, self.val_losses = [], []
        self.train_metrics = []  
        self.val_metrics = []    
        self.best_epoch = None

        Variables().confidence_avg = self.confidence_avg

    def _get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(grouped_params, lr=self.config.lr, eps=self.config.adam_eps)

    def loss_fn(self, logits, labels):
        weights = torch.tensor(self.config.tag_weight, device=self.accelerator.device)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def prepare(self):
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0
        all_preds, all_labels = [], [] 
        for step, inputs in enumerate(tqdm(self.train_loader, desc=f"Train Epoch {epoch}")):
            subset = {k: inputs[k] for k in ['input_ids', 'attention_mask'] if k in inputs}
            with self.accelerator.accumulate(self.model):
                outputs = self.model(subset)
                loss = self.loss_fn(outputs, inputs['labels'])
                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.config.enable_scheduler:
                    self.scheduler.step(epoch - 1 + step / len(self.train_loader))
                self.optimizer.zero_grad()
                running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=-1)
            labels = inputs['labels']
            mask = labels != -100  # -100인 토큰은 무시
            all_preds.extend(preds[mask].detach().cpu().tolist())
            all_labels.extend(labels[mask].detach().cpu().tolist())

        # 평균 손실 계산
        self.train_losses.append(running_loss / len(self.train_loader))

        # 정확도, 정밀도, 재현율, F1 스코어 계산
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        self.train_metrics.append({
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0
        all_preds, all_labels = [], []

        for inputs in tqdm(self.val_loader, desc=f"Valid Epoch {epoch}"):
            subset = {k: inputs[k] for k in ['input_ids', 'attention_mask'] if k in inputs}
            outputs = self.model(subset)
            loss = self.loss_fn(outputs, inputs['labels'])
            running_loss += loss.item()

            # 분류 지표를 위한 예측값/정답 수집
            preds = torch.argmax(outputs, dim=-1)
            labels = inputs['labels']
            mask = labels != -100
            all_preds.extend(preds[mask].detach().cpu().tolist())
            all_labels.extend(labels[mask].detach().cpu().tolist())

        self.val_losses.append(running_loss / len(self.val_loader))
        # 정확도, 정밀도, 재현율, F1 스코어 계산
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        self.val_metrics.append({
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        # 첫 에폭에서 confidence 평균 계산
        if epoch == 1:
            self.confidence_avg = sum([float(max(outputs[0][m])) for m in range(inputs['attention_mask'].sum().item())]) / inputs['attention_mask'].sum().item()

    def fit(self, output_dir: str):
        self.prepare()
        best_val_loss = float('inf')
        for epoch in range(1, self.config.epochs + 1):
            self.train_one_epoch(epoch)
            self.valid_one_epoch(epoch)
            print(f"Epoch {epoch} | Train Loss: {self.train_losses[-1]:.4f} | Val Loss: {self.val_losses[-1]:.4f} | Accuracy: {self.val_metrics[-1]}")
            if self.val_losses[-1] < best_val_loss:
                best_val_loss = self.val_losses[-1]
                self.best_epoch = epoch
                self.accelerator.save(self.model.state_dict(), f"{output_dir}/clause_model_earth.pt")
            gc.collect()
            torch.cuda.empty_cache()
    
    def save_metrics(self, output_dir: str):
            """
            훈련 및 검증 손실과 분류 지표를 metrics.json으로 저장합니다.
            """
            os.makedirs(output_dir, exist_ok=True)
            metrics = {
                'best_epoch': self.best_epoch,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics
            }
            with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            print(f"Metrics saved to {os.path.join(output_dir, 'metrics.json')}")
    
    def plot_metrics(trainer, save_dir):
        """
        Trainer 인스턴스가 가지고 있는 train/val 손실과 분류 지표를
        에폭별로 그래프로 저장합니다.
        """
        os.makedirs(save_dir, exist_ok=True)  # 저장 폴더가 없으면 생성

        # 1) 손실(loss) 그리기
        plt.figure()
        # train_losses와 val_losses는 리스트이므로 그대로 사용
        plt.plot(trainer.train_losses, label='train_loss')
        plt.plot(trainer.val_losses,   label='val_loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.close()

        # 2) 분류 지표(accuracy, precision, recall, f1) 그리기
        # trainer.train_metrics, trainer.val_metrics는 [{...}, {...}, …] 꼴이므로
        # metric별로 리스트를 뽑아낸 후 그립니다.
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.figure()
            # 각 epoch마다 저장된 metric 값을 리스트로 추출
            train_vals = [m[metric] for m in trainer.train_metrics]
            val_vals   = [m[metric] for m in trainer.val_metrics]
            plt.plot(train_vals, label=f'train_{metric}')
            plt.plot(val_vals,   label=f'val_{metric}')
            plt.title(f'{metric.capitalize()} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{metric}.png'))
            plt.close()


# ------------------------------
# 실행 함수
# ------------------------------
def main():
    config = Config()
    label_data = LabelData()

    # 모델 설정
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = DebertaV2ForTokenClassification.from_pretrained(
        config.model,
        num_labels=4,
        id2label=label_data.id2label,
        label2id=label_data.label2id
    )

    # 데이터셋 로드 및 분할
    df = open_file('Etaging.txt')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = TokenTaggingDataset(train_df, config, tokenizer, max_len=config.max_length)
    val_ds = TokenTaggingDataset(val_df, config, tokenizer, max_len=config.max_length)

    # DataLoader 생성
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # 학습 시작
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    model = TaggingModel(config)
    trainer = Trainer(model, (train_loader, val_loader), config, accelerator)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'save/{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    trainer.fit(save_dir)
    trainer.save_metrics(save_dir)
    trainer.plot_metrics(save_dir)

if __name__ == "__main__":
    main()
