"""
clause_splitting.py

📌 목적:
    문장을 절(clause) 단위로 분할하고, 각 절의 BERT 계열 임베딩을 생성하며,
    중요 단어를 식별하여 하이라이팅 정보를 생성합니다.

🔧 주요 기능:
    - 문장 분할: 미리 학습된 TaggingModel을 통해 문장을 의미 단위 절로 분할
    - 임베딩 생성: 각 절을 DeBERTa 기반 모델로 인코딩하여 CLS 벡터 추출
    - 중요 단어 추출: 각 단어 벡터와 CLS 벡터 간 cosine 유사도로 단어 중요도 평가
    - 하이라이팅: 중요 단어 기반으로 시각적 강조 정보 생성
    - 출력 파일 생성:
        - `splited.json`: 문장 → 절 변환 결과 저장
        - `clause_embedding.npy`: 각 절의 임베딩 벡터 저장 (NumPy array)
        - `significant.jsonl`: 절별 중요 단어 리스트 저장

📁 입력:
    - `example2.txt`: 문장 목록 (줄마다 문장 하나)

📁 출력:
    - `./saved_temp/embedding_batch_*.npy`: 임시 저장된 임베딩 파일들
    - `clause_embedding.npy`: 전체 병합된 임베딩 벡터
    - `splited.json`: 문장을 절 단위로 분할한 결과
    - `significant.jsonl`: 절별 중요 단어 리스트

⚙️ 클래스:
    - `ClauseSpliting`: 전체 파이프라인을 캡슐화한 메인 클래스
        - split2Clause(): 절 분할 수행
        - clause_embedding(): 임베딩 생성 및 중요 단어 식별
        - highlight_jsonl(): 생성된 JSONL을 통해 하이라이트 문장 출력
        - 내부적으로 KIWI 형태소 분석기 및 HuggingFace Transformers 사용

🎯 사용 예시:
    ```bash
    python clause_splitting.py
    ```

📝 요구사항:
    - pretrained TaggingModel (`clause_model_earth.pt`)
    - huggingface model (`kakaobank/kf-deberta-base`)
    - KIWI, tqdm, torch, numpy, transformers 등

"""

from transformers import AutoTokenizer, AutoModel, DebertaV2Model
import torch
import torch.nn.functional as F
from kiwipiepy import Kiwi
from tqdm import tqdm
from train import Config, Variables, TaggingModel, LabelData
from typing import Literal
import numpy as np
import json
from dataclasses import dataclass
import os

@dataclass
class FileNames():
    clause_model_pt : str = "clause_model_earth.pt"
    splited_json : str    = 'splited.json'
    embedding_np : str    = 'clause_embedding.npy'
    significant_json: str = 'significant.jsonl'
    saved_temp_dir : str  = './saved_temp'

@torch.no_grad()
def prediction(model, tokenizer, sentence, label_map, device='cuda', max_length=128, return_cls = False):
    """
    문장을 입력 받아 토큰 단위 BIO 태깅 결과와 신뢰도를 반환합니다.
    선택적으로 문장 전체의 CLS 벡터도 함께 반환할 수 있습니다.

    Args:
        model (nn.Module): 토큰 분류 모델
        tokenizer (PreTrainedTokenizer): 해당 모델의 토크나이저
        sentence (str): 입력 문장
        label_map (dict): 레이블 인덱스 → 이름 매핑
        device (str): 실행 디바이스 ('cuda' 또는 'cpu')
        max_length (int): 최대 토큰 길이
        return_cls (bool): True일 경우 CLS 벡터도 반환

    Returns:
        list of (token, label, confidence) 또는 (list, torch.Tensor):
            return_cls=True일 경우 CLS 벡터 포함 튜플 반환

    Example:
        >>> prediction(model, tokenizer, "금리가 오르면 대출이 줄어든다.", label_map)
        [('금리', 'O', 0.99), ('가', 'O', 0.98), ..., ('다.', 'E3', 0.96)]
    """
    model.eval()
    model.to(device)

    # 토크나이즈 및 입력 구성
    encoding = tokenizer(
        sentence,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_attention_mask=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0]  # (L, 2)

    outputs, cls_vector= model({'input_ids': input_ids, 'attention_mask': attention_mask}, return_cls=True)
    confidences = [float(int(float(max(m))*10000)/10000) for m in outputs[0]]
    preds = torch.argmax(outputs, dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    results = []
    for i, (token, pred, confidence, offset) in enumerate(zip(tokens, preds, confidences, offset_mapping)):
        if offset[0].item() == 0 and offset[1].item() == 0:
            continue  # [PAD] 토큰 제외
        results.append((token, label_map[pred], confidence))
    if return_cls:
        results = (results, cls_vector)

    return results

def recover_wordpieces(tokens: list) -> str :
    """
    WordPiece 토큰 리스트를 원래 단어 문자열로 복원합니다.

    Args:
        tokens (list): BERT-style WordPiece 토큰 리스트

    Returns:
        str: 복원된 문장 문자열

    Example:
        >>> recover_wordpieces(["금", "##리", "가", "상", "##승", "했", "##다"])
        '금리가 상승했다'
    """
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
    return ' '.join(words)

def highlight(sentences: list[list[str]], highlight_words: list[list[list[str]]]) -> str:
    """
    중요 단어들을 하이라이트 처리하여 텍스트 형태로 반환합니다.

    Args:
        sentences (list): 절 단위로 분할된 문장 리스트
        highlight_words (list): 절별 중요 단어 리스트

    Returns:
        str: ANSI 색상 코드로 하이라이트된 문장

    Example:
        >>> highlight([["금리가 오르면", "대출이 줄어든다"]],
                     [[["금리"], ["대출"]]])
        '\033[95m금리\033[0m가 오르면 / \033[95m대출\033[0m이 줄어든다'
    """
    color = ('\033[95m', '\033[0m')
    highlighted_sentences = []

    def split_by_keyword(text: str, keyword: str):
        idx = text.find(keyword)
        return [text[:idx], keyword, text[idx + len(keyword):]] if idx != -1 else []

    for clause_list, clause_keywords in zip(sentences, highlight_words):
        # 구문(절) 기준으로 강조 적용
        highlighted_clauses = []
        for clause, keywords in zip(clause_list, clause_keywords):
            result = []
            for word in clause.split():
                q = sum([split_by_keyword(word, term) for term in keywords], [])
                if q:
                    result.append(f"{q[0]}{color[0]}{q[1]}{color[1]}{q[2]}")
                else:
                    result.append(word)
            highlighted_clauses.append(' '.join(result))
        # 슬래시로 구분
        highlighted_sentence = ' / '.join(highlighted_clauses)
        highlighted_sentences.append(highlighted_sentence)

    return '\n'.join(highlighted_sentences)

def highlight_jsonl(jsonl_path: str):
    sentences, highlight_words = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sentences.append(item["clause"])
            highlight_words.append(item["highlight"])
    return highlight(sentences, highlight_words)

class ClauseSpliting():
    def __init__(self, sentences, e_option: Literal['all','E3','E2','E'] = 'E3', threshold = True):
        self.kiwi = Kiwi()
        self.config = Config()
        self.config.save_batch = 100
        self.config.return_embed_max = 200
        self.model = TaggingModel(self.config)
        self.model.load_state_dict(torch.load(FileNames().clause_model_pt))
        self.embedding_model = DebertaV2Model.from_pretrained(self.config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.sentences = sentences
        self.cls_vectors = []
        option_map = {
            'all': ['E', 'E2', 'E3'],
            'E2': ['E', 'E2'],
            'E3': ['E', 'E3'],
            'E': ['E']}
        self.elist = option_map.get(e_option, ['E'])
        self.threshold = Variables().confidence_avg * self.config.confidence_threshold if threshold else 0.0
        self.splited = self.split2Clause()
        with open(FileNames().splited_json, "w", encoding="utf-8-sig") as f:
            json.dump(self.splited, f, ensure_ascii=False, indent=2)
        self.embeds = self.clause_embedding(self.splited)

    def split2Clause(self):
        """
        입력된 문장을 절(clause) 단위로 분할합니다.

        Returns:
            list: 절 단위로 분할된 문장 리스트 (리스트 of 리스트)

        Example:
            >>> splitter = ClauseSpliting("금리가 오르면 대출이 줄어든다.")
            >>> splitter.split2Clause()
            [["금리가 오르면", "대출이 줄어든다"]]
        """
        if isinstance(self.sentences, str):
            _sentences = [self.sentences]
        else:
            _sentences = self.sentences
        results = []
        for sentence in tqdm(_sentences):
            predicted = prediction(self.model, self.tokenizer, sentence, LabelData().id2label, return_cls=True)
            self.cls_vectors.append(predicted[-1])
            predicted = predicted[0]
            clauses, clause, switch = [], [], False
            for i, (tok, label, confidence) in enumerate(predicted):
                if label in self.elist and confidence > self.threshold and not self.is_segm(tok, predicted[i][0]):
                    switch = True
                elif switch:
                    recovered = recover_wordpieces(clause)
                    if len(recovered.split()) < 2 and clauses:
                        clauses[-1] += ' ' + recovered.strip()
                    else:
                        clauses.append(recovered)
                    clause, switch = [], False
                clause.append(tok)
            if clause:
                clauses.append(recover_wordpieces(clause))
            results.append(clauses)
        return results if not isinstance(self.sentences, str) else results[0]

    def clause_embedding(self, splited):
        """
        절 단위로 BERT 기반 임베딩을 추출하고, 중요 단어(highlight)를 계산합니다.

        Args:
            splited (list): 절 분할 결과

        Returns:
            list: 각 절에 대한 CLS 벡터 리스트

        Side Effects:
            - 'clause_embedding.npy': 절 임베딩 저장
            - 'significant.jsonl': 중요 단어 JSON 저장
        """
        def save_batch_npy(batch_result, save_dir, batch_idx):
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'embedding_batch_{batch_idx}.npy')
            np.save(path, np.array(batch_result, dtype=object))  # allow_pickle=True 필요

        with open(FileNames().significant_json, "w", encoding="utf-8") as f:
            pass

        all_result = [] if len(splited) < self.config.return_embed_max else None
        for batch_idx in range(0, len(splited), self.config.save_batch):
            batch = splited[batch_idx:batch_idx + self.config.save_batch]
            result, highlighted = [], []
            for ss in tqdm(batch, desc=f"Batch {batch_idx // self.config.save_batch}"):
                temp, highlight_temp = [], []
                for s in ss:
                    inputs = self.tokenizer(s, return_tensors='pt', add_special_tokens=True)
                    input_ids = inputs["input_ids"]
                    with torch.no_grad():
                        outputs = self.embedding_model(**inputs)
                    hidden_states, cls_vector = outputs.last_hidden_state, outputs.last_hidden_state[:, 0, :]
                    temp.append(cls_vector.squeeze(0).cpu().numpy())
                    real = self.str2real(s, output_str=False)
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                    token_map = []
                    for idx, tok in enumerate(tokens):
                        if tok in self.tokenizer.all_special_tokens:
                            continue
                        clean_tok = tok[2:] if tok.startswith("##") else tok
                        for word in real:
                            if clean_tok in word:
                                token_map.append((word, idx))
                                break
                    word2indices = {}
                    for word, idx in token_map:
                        word2indices.setdefault(word, []).append(idx)
                    word_scores = []
                    for word, indices in word2indices.items():
                        vecs = torch.stack([hidden_states[0, i] for i in indices])
                        sims = F.cosine_similarity(vecs, cls_vector[0].unsqueeze(0), dim=1)
                        score = self.rms(sims)
                        word_scores.append((word, float(score)))
                    word_scores_sorted = sorted(word_scores, key=lambda x: x[1], reverse=True)
                    top_n = max(1, int(len(word_scores_sorted) * 0.6))
                    top_words = {word for word, _ in word_scores_sorted[:top_n]}
                    highlight_temp.append([word for word in real if word in top_words])
                highlighted.append(highlight_temp)
                result.append(temp)

            # save
            for clauses, highlights in zip(batch, highlighted):
                item = {"clause"   : clauses,
                        "highlight": highlights}
                with open(FileNames().significant_json, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            save_batch_npy(result, FileNames().saved_temp_dir, batch_idx) # 복원시 allow_pickle=True 옵션 필수

            if all_result is not None:
                all_result.extend(result)

        def load_and_merge_npy(save_dir: str, output_path: str):
            files = [f for f in os.listdir(save_dir) if f.startswith("embedding_batch_") and f.endswith(".npy")]
            if not files:
                raise FileNotFoundError("병합할 .npy 파일이 없습니다.")
            files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(files) == 1:
                src = os.path.join(save_dir, files[0])
                dst = os.path.join(save_dir, output_path)
                np.save(dst, np.load(src, allow_pickle=True))
                return
            merged = []
            for file in files:
                batch_path = os.path.join(save_dir, file)
                data = np.load(batch_path, allow_pickle=True)
                merged.extend(data)
            np.save(output_path, np.array(merged, dtype=object))

        # 임시 저장된 파일 병합
        load_and_merge_npy(FileNames().saved_temp_dir,FileNames().embedding_np)
        return all_result

    def is_gram(self, word):
        """조사/어미/접사 여부를 판별합니다."""
        t = self.kiwi.tokenize(word)[-1].tag
        return t[0] in ['J', 'E'] or t[:2] == 'XS'

    def is_segm(self, word, prev):
        """절 분리 가능 여부를 판별합니다."""
        combined = prev + word.strip('#') if word.startswith('#') else prev + ' ' + word
        t = self.kiwi.tokenize(combined)[-1].tag
        return t[0] in ['N', 'V', 'M'] or t[:2] == 'XR'

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        """Root Mean Square 계산 함수"""
        return torch.sqrt(torch.mean(x ** 2))

    def str2real(self, text, timecat=True, output_str=True):
        """
        입력 문장에서 실질 단어만 추출합니다 (명사/동사 등).

        Args:
            text (str): 원본 문장
            timecat (bool): 연속된 시간 관련 단어를 하나로 묶을지 여부
            output_str (bool): 문자열로 반환할지 여부

        Returns:
            list or str: 실질 단어 리스트 또는 문자열
        """
        tokens = self.kiwi.tokenize(text)
        return ' '.join(self.bereal(tokens, timecat)) if output_str else self.bereal(tokens, timecat)

    def bereal(self, tokens, timecat=True):
        """형태소 토큰 중 의미 있는 단어만 추출"""
        real, timeset = [], []
        take = ['NNG', 'NNP', 'NNB', 'NP', 'NR', 'XR', 'SN', 'SL', 'VV', 'VA', 'MM', 'MAJ', 'MAG']
        timeTrigger = ['년', '월', '일', '시', '분', '초', '세']
        for token in tokens:
            if token.tag in take:
                if not timecat:
                    real.append(token.form)
                    continue
                if token.tag in ['SN', 'NR']:
                    timeset.append(token.form)
                elif token.form in timeTrigger or token.tag == 'NNB':
                    if timeset:
                        timeset.append(token.form)
                elif len(timeset) > 1:
                    real.append(''.join(timeset))
                    timeset = []
                elif timeset:
                    real.append(timeset[0])
                    timeset = []
                real.append(token.form)
        return real

def main():
    config = Config()
    config.confidence_threshold = 0.15

    with open('example2.txt', 'r', encoding='utf-8-sig') as f:
        raw = f.read()
        sentences = [r for r in raw.splitlines()]

    r = ClauseSpliting(sentences, e_option= 'E3', threshold= True)


if __name__ == "__main__":
    main()