from transformers import DebertaV2ForTokenClassification, AutoTokenizer, DebertaV2Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
from dataclasses import dataclass, field

# Config
@dataclass
class Config:
    model: str = "kakaobank/kf-deberta-base"
    dropout: float = 0.5
    max_length: int = 128
    batch_size: int = 1
    epochs: int = 50
    lr: float = 3e-4
    enable_scheduler: bool = True
    scheduler: str = 'CosineAnnealingWarmRestarts'
    gradient_accumulation_steps: int = 2
    adam_eps: float = 1e-6
    freeze_encoder: bool = True
    tag_weight: list = field(default_factory=lambda: [0.1, 1.0, 1.2, 1.2])
    confidence_threshold: float = 0.5
@dataclass
class LabelData:
    labels: list = field(default_factory=lambda: ["O", "E", "E2", "E3"])
    id2label: dict = field(init=False)
    label2id: dict = field(init=False)

    def __post_init__(self):
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}
@dataclass
class Variables:
    confidence_avg : float = 1.0

# tokens -> sentence
def recover_wordpieces(tokens: list) -> str :
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

# open file
def open_file(file_name):
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        raw = f.read()
        result = []
        sents = []
        tags = []
        for r in raw.splitlines():
            r = r.strip()
            if len(r) > 0:
                rr = r.split()
                if len(rr) != 2:
                    print("nonononononnonoono")
                sents.append(rr[0])
                tags.append(rr[1])
            else:
                result.append({'tokens':sents, 'labels':tags})
                sents = []
                tags = []

    print("The number of data sentences : ",len(result))

    for r in result:
        tokens = r["tokens"]
        sentence = recover_wordpieces(tokens)
        r['full_text'] = sentence

    result = result[:166] # 166번까지가 가장 효율이 좋음음
    return pd.DataFrame(result)

# Dataset
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
            padding="max_length",  # or True
            max_length=self.max_len,
            return_tensors='pt'
        )
        iter_labels = iter(labels)
        label_ids = []
        for id in encoding['input_ids'].squeeze():
            if id < 6: # kf-deberta
                label_ids.append(-100)
            elif id >= 130000: # kf-deberta
                label_ids.append(-100)
            else:
                # label_ids.append(label2id[next(iter_labels)])
                try:
                    label = next(iter_labels)
                    label_id = self.label2id[label]
                    if not 0 <= label_id < len(self.label2id):
                        print("잘못된 label:", label, "→", label_id)
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

# Pooling
class MeanPooling(nn.Module):
    def forward(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embed = torch.sum(hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return sum_embed / sum_mask

# Model
class TaggingModel(nn.Module):
    def __init__(self, config, num_classes=4):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(config.model, output_hidden_states=True)
        if config.freeze_encoder:
            for p in self.encoder.base_model.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)  # [hidden → num_classes]

    def forward(self, inputs, return_cls=False, out_last_hidden_state = False):
        out = self.encoder(**inputs, return_dict=True)
        sequence_output = self.dropout(out.last_hidden_state)  # shape: [B, L, H]
        logits = self.classifier(sequence_output)              # shape: [B, L, C]
        result = [logits]
        if return_cls:
            cls_vector = sequence_output[:, 0, :]               # [CLS] 벡터
            result.append(cls_vector)
        if out_last_hidden_state:
            result.append(out.last_hidden_state)
        return result if any((return_cls,out_last_hidden_state)) else logits
    
# Trainer
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
        # for inputs, targets in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):
        for step, inputs in enumerate(tqdm(self.train_loader, desc=f"Train Epoch {epoch}")):
            subset = {k: inputs[k] for k in ['input_ids','attention_mask'] if k in inputs}
            with self.accelerator.accumulate(self.model):
                outputs = self.model(subset)
                loss = self.loss_fn(outputs, inputs['labels'])
                # loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.config.enable_scheduler:
                    self.scheduler.step(epoch - 1 + step / len(self.train_loader))
                self.optimizer.zero_grad()
                running_loss += loss.item()
        self.train_losses.append(running_loss / len(self.train_loader))

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0
        for inputs in tqdm(self.val_loader, desc=f"Valid Epoch {epoch}"):
            subset = {k: inputs[k] for k in ['input_ids','attention_mask'] if k in inputs}
            outputs = self.model(subset)
            loss = self.loss_fn(outputs, inputs['labels'])
            running_loss += loss.item()
        self.val_losses.append(running_loss / len(self.val_loader))
        if epoch == 1:
            self.confidence_avg = sum([float(max(outputs[0][m])) for m in range(inputs['attention_mask'].sum().item())])/inputs['attention_mask'].sum().item()

    def fit(self):
        self.prepare()
        best_val_loss = float('inf')
        for epoch in range(1, self.config.epochs + 1):
            self.train_one_epoch(epoch)
            self.valid_one_epoch(epoch)
            print(f"Epoch {epoch} | Train Loss: {self.train_losses[-1]:.4f} | Val Loss: {self.val_losses[-1]:.4f}")
            if self.val_losses[-1] < best_val_loss:
                best_val_loss = self.val_losses[-1]
                self.accelerator.save(self.model.state_dict(), "clause_model_earth.pt")
            gc.collect()
            torch.cuda.empty_cache()

def main():
    config = Config()
    label_data = LabelData()

    # model setting
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = DebertaV2ForTokenClassification.from_pretrained(
        config.model,
        num_labels=4,
        id2label = label_data.id2label,
        label2id = label_data.label2id
    )

    # Dataset split
    df = open_file('Etaging.txt')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = TokenTaggingDataset(train_df, config, tokenizer, max_len=config.max_length)
    val_ds = TokenTaggingDataset(val_df, config, tokenizer, max_len=config.max_length)

    # data load
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True) # num_workers=1 
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False) # num_workers=1

    # Train
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    model = TaggingModel(config)
    trainer = Trainer(model, (train_loader, val_loader), config, accelerator)
    trainer.fit()


if __name__ == "__main__":
    main()