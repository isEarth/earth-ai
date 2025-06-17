# 전체 시스템은 다음과 같은 목적을 갖는다:
# - 입력 문장을 절(clause) 단위로 분리하고
# - 각 절의 의미 임베딩 벡터와 의미 강조 단어를 추출하며
# - 절 간의 의미적 관계(triplet)을 추출해 Knowledge Graph로 확장 가능하도록 지원함

# === 주요 모듈 ===
# - ClauseSpliting : 문장 -> 절 분리 + 임베딩 + 강조 단어 추출 + 관계 추출
# - ClauseDB       : SQLite DB 관리 및 임베딩 조회
# - prediction     : tagging model을 통해 문장에서 E/E2/E3 위치 예측
# - highlight      : 의미 강조 단어를 시각화 출력

# ========================
# 주요 클래스: ClauseSpliting
# ========================
# - 문장 리스트를 입력으로 받아 다음 단계 처리:
#   1. split2Clause      : tagging model로 절 분리
#   2. clause_embedding  : 각 절마다 [CLS] 임베딩 및 중요 단어 추출
#   3. find_rel          : 절 간 관계 triplet 구성
#   4. set_db            : 절을 clause_id 기반으로 DB에 저장
#   5. splited_id_mapping: DB에서 id 포함된 절 구조 반환
#   6. print_triplets    : 관계 출력
#   7. summary           : 전체 분석 요약 출력

# ======================
# 주요 구조
# ======================
# clause_id = V*100000 + S*10 + C
#   - V: video index
#   - S: sentence index within video
#   - C: clause index within sentence

# self.splited : [video][sentence][clause] 
# self.embeds  : [video][sentence][clause][768] -> 각 clause의 CLS 임베딩 벡터
# self.meanpooled_embeds: 각 토큰의 임베딩을 clause 단위로 mean한 임베딩

# ======================
# 관계 추출 기준
# ======================
# - 어미(aumi), 접속사(conj), 어간(augan)에 따라 문장의 마지막이나 처음을 보고 단서를 추출
# - 역할마다 우선순위가 달라서 먼저 발견되는 단서를 사용
# - 관계가 중첩되는 경우(앞/뒤) max()로 덮어씀

# ======================
# 주요 설정값
# ======================
# config.confidence_threshold  : tagging 확률 threshold (절 분리 민감도)
# config.important_words_ratio : cosine similarity 상위 몇 % 단어를 강조로 볼 것인가
# config.clause_len_threshold  : 절 최소 길이 제한

# ======================
# 실행 진입점
# ======================
# main():
#   - youtube_filtered.json이 있으면 전처리 생략
#   - ClauseSpliting 객체를 생성해 전체 파이프라인 실행 (분리, 임베딩, 관계 추출)
#   - 결과를 요약 및 출력

# ======================
# 기타 함수
# ======================
# - highlight_jsonl: 강조 단어와 절을 JSONL에서 불러와 콘솔에 컬러 출력
# - recover_wordpieces: WordPiece 토큰을 원래 단어로 복원
# - get_shape: nested list / tensor의 shape 확인용 디버깅 함수
# - bereal: 형태소 분석 후 의미 요소만 필터링 (숫자, 명사, 동사 등)

# ======================


from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from kiwipiepy import Kiwi
from tqdm import tqdm
import numpy as np
import pandas as pd
from train import Config, Variables, TaggingModel, LabelData
from processing import open_and_preprocess, select_terms
from typing import Literal
import json
from dataclasses import dataclass
import os
import sqlite3


@dataclass
class FileNames():
    clause_model_pt : str = "../clause_model_earth.pt"
    # --------------- #
    extra_name : str      = "_top6"
    saved_dir : str       = "./saved_data/"
    splited_json : str    = saved_dir+ f'splited{extra_name}.json'
    embedding_np : str    = saved_dir+ f'clause_embedding{extra_name}.npy'
    sbert_np : str        = saved_dir+ f'sbert{extra_name}.npy'
    significant_jsonl:str = saved_dir+ f'significant{extra_name}.jsonl'
    clause_db: str        = saved_dir+ f'clause{extra_name}.db'
    triplets_np: str      = saved_dir+ f'triplets{extra_name}.npy'
    saved_temp_dir : str  = './saved_temp'
    relation_trigger: str = "../data/relation_trigger.csv"

@torch.no_grad()
def prediction(model, tokenizer, sentence, label_map, device='cuda', max_length=128, return_cls=False, return_lhs = False):
    """
    문장을 입력받아 tagging model을 통해 각 토큰의 label과 confidence를 예측합니다.

    Args:
        model (nn.Module): 학습된 시퀀스 태깅 모델
        tokenizer (PreTrainedTokenizer): 해당 모델에 맞는 tokenizer
        sentence (str): 입력 문장
        label_map (dict): 예측 결과 ID를 라벨(str)로 변환하기 위한 매핑
        device (str): 연산에 사용할 디바이스 ('cuda' 또는 'cpu')
        max_length (int): 입력 문장의 최대 토큰 길이
        return_cls (bool): [CLS] 벡터를 반환할지 여부
        return_lhs (bool): 마지막 hidden state 전체를 반환할지 여부

    Returns:
        - return_cls=False: List[Tuple[token, label, confidence]]
        - return_cls=True: Tuple[위 리스트, cls_vector] 또는 [추가로 hidden state] 포함
    """

    # gpu에 모델이 없다면 올리기 
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()

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

    outputs, cls_vector, last_hidden_state= model({'input_ids': input_ids, 'attention_mask': attention_mask}, return_cls=True, return_last_hidden_state=True)
    confidences = [float(int(float(max(m)) * 10000) / 10000) for m in outputs[0]]  # 각 토큰의 confidence 정규화
    preds = torch.argmax(outputs, dim=-1)[0].cpu().tolist()  # 가장 높은 점수의 클래스 예측
    
    valid_len = attention_mask.sum().item()  # 실제 토큰 수
    masked_lhs = last_hidden_state.squeeze(0)[:valid_len]  # shape: [S, H]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    be_clause = []
    results = []
    for token, pred, confidence, offset in zip(tokens, preds, confidences, offset_mapping):
        if offset[0].item() == 0 and offset[1].item() == 0:
            continue  # [PAD] 토큰 제외
        be_clause.append((token, label_map[pred], confidence))
    if return_cls:
        results.append(be_clause)
        results.append(cls_vector.detach().cpu())
    if return_lhs:
        results.append(masked_lhs.detach().cpu())
        
    return tuple(results) if return_cls else be_clause

def recover_wordpieces(tokens: list) -> str:
    """
    WordPiece 토큰 리스트를 원래 단어로 복원합니다.

    Args:
        tokens (List[str]): WordPiece로 분할된 토큰 리스트

    Returns:
        str: 공백 기준으로 병합된 문자열 (예: ['국', '##민', '은행'] → '국민 은행')
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

def highlight(sentences: list[list[str]], highlight_words: list[list[list[str]]],return_in_list = False) -> str:
    """
    절과 강조 단어 리스트를 받아, 특정 단어를 ANSI 색상으로 강조한 문자열을 생성합니다.

    Args:
        sentences (List[List[str]]): 절 리스트, ex) [[절1, 절2], ...]
        highlight_words (List[List[List[str]]]): 강조할 단어 리스트 (각 절 단위)
        return_in_list (bool): True이면 리스트 반환, False이면 줄바꿈 문자열 반환

    Returns:
        str or List[str]: 강조 단어가 색칠된 문장
    """

    # print("highlight : ",get_shape(sentences), get_shape(highlight_words))

    color = ('\033[95m', '\033[0m')  # ANSI 콘솔 마젠타
    highlighted_sentences = []

    def split_by_keyword(text: str, keyword: str):
        idx = text.find(keyword)
        return [text[:idx], keyword, text[idx + len(keyword):]] if idx != -1 else []

    for clause_list, clause_keywords in zip(sentences, highlight_words):
        highlighted_clauses = []
        for clause, keywords in zip(clause_list, clause_keywords):
            result = []
            for word in clause.split():
                q = sum([split_by_keyword(word, term) for term in keywords], [])
                if q and type(q)==list:
                    result.append(f"{q[0]}{color[0]}{q[1]}{color[1]}{q[2]}")
                else:
                    result.append(word)
            highlighted_clauses.append(' '.join(result))
        highlighted_sentence = ' / '.join(highlighted_clauses)
        highlighted_sentences.append(highlighted_sentence)

    return highlighted_sentences if return_in_list else '\n'.join(highlighted_sentences)

def highlight_jsonl(jsonl_path: str, sample: int =float('inf'))->list:
    """
    강조 단어가 포함된 JSONL 파일을 불러와 콘솔용 문자열로 변환합니다.

    Args:
        jsonl_path (str): JSONL 파일 경로
        sample (int): 최대 출력할 문장 수

    Returns:
        List[str]: 강조 문장 문자열 리스트
    """
    videos, highlight_words = [], []
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            videos.append(item["clause"])
            highlight_words.append(item["highlight"])
            if i > sample:
                break
    return [highlight(sents, highlight_sents) for sents, highlight_sents in zip(videos, highlight_words)]

def get_shape(obj):
    """
    입력 객체의 재귀적 shape 정보를 반환합니다.

    Args:
        obj (Union[list, np.ndarray, torch.Tensor]): 대상 객체

    Returns:
        Tuple[int, ...]: 객체의 다차원 shape
    """
    # 텐서나 넘파이 배열이면 바로 shape 반환
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return tuple(obj.shape)

    # 기본형 (숫자, 문자열 등) -> shape 없음
    if not isinstance(obj, list):
        return ()

    # 빈 리스트인 경우
    if len(obj) == 0:
        return (0,)

    # 리스트일 경우 재귀
    first_shape = get_shape(obj[0])
    return (len(obj),) + first_shape


class ClauseSpliting:
    """
    문장을 절(clause) 단위로 분리하고 각 절의 의미 임베딩, 의미 강조 단어, 그리고 절 간 의미 관계를 추출하는 통합 처리 클래스.

    주된 기능:
    - 구문 분리: 문장을 tagging model로 분석하여 E/E2/E3 태그 기반으로 절을 분리함
    - 임베딩 생성: 각 절에 대해 DeBERTa 모델을 사용하여 [CLS] 벡터 및 mean pooling 벡터 생성
    - 의미 강조 단어 추출: [CLS] 벡터와 토큰 간 cosine 유사도 기반으로 중요한 단어 선정
    - 관계 추출: 단서사전 기반으로 절 간 인과/대조/조건 등의 의미 관계(triplet)를 추출
    - DB 저장: 각 절을 고유 ID로 구성된 DB에 저장하며, 임베딩도 numpy 파일로 저장 가능

    주요 입력:
    - sentences: [str] 또는 [List[List[str]]] 형태의 문장 집합
    - config: 모델 및 임계값 설정을 담은 Config 객체
    - filenames: 파일 경로를 담은 FileNames 객체

    주요 출력/저장:
    - self.splited: 절 단위 분할 결과 [[video][sentence][clause]]
    - self.meanpooled_embeds: 절별 mean pooling 벡터
    - self.embeds: DeBERTa 기반 절 임베딩 ([CLS])
    - ./saved_data 디렉토리에 JSON, JSONL, NPY, DB 등 다수 파일 저장
    """
    def __init__(self, sentences = None, config = Config(), filenames =FileNames(), e_option: Literal['all', 'E3', 'E2', 'E'] = 'E3', threshold=True, reference_mode = False):
        """
        초기화 메서드
        - tokenizer, tagging model, embedding model, 설정 값 등을 로드함
        - 절 분리 수행 및 결과를 JSON 파일로 저장
        - 각 절에 대해 임베딩 수행 및 의미 강조 단어 추출
        """
        self.kiwi = Kiwi()
        self.filenames = filenames
        os.makedirs(filenames.saved_dir, exist_ok=True)
        self.history = {'num_triplets': 0, 'num_clauses': 0, 'num_sentences': 0, 'num_videos': 0}
        self.config = config
        self.config.clause_len_threshold = getattr(self.config, 'clause_len_threshold', 3)
        self.config.save_batch = getattr(self.config, 'save_batch', 100)
        self.config.return_embed_max = getattr(self.config, 'return_embed_max', 200)
        self.config.important_words_ratio = getattr(self.config, 'important_words_ratio', 0.6)
        self.model = TaggingModel(self.config)
        self.model.load_state_dict(torch.load(self.filenames.clause_model_pt))
        self.embedding_model = AutoModel.from_pretrained(self.config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.concat_project = ConcatProject()
        self.sentences = sentences
        self.meanpooled_embeds = None
        self.cls_vectors = []
        self.rel_map = {'없음': 0, '기타': 1, '대조/병렬': 2, '상황': 3, '수단': 4, '역인과': 5, '예시': 6, '인과': 7}
        option_map = {'all': ['E', 'E2', 'E3'], 'E2': ['E', 'E2'], 'E3': ['E', 'E3'], 'E': ['E']}
        self.elist = option_map.get(e_option, ['E'])
        self.threshold = Variables().confidence_avg * self.config.confidence_threshold if threshold else 0.0
        if (not reference_mode) and (not sentences) :
            raise ValueError("sentences must be exist in reference_mode=False")

        if not reference_mode:
            switch = False
            if os.path.exists(self.filenames.splited_json):
                with open(self.filenames.splited_json, "r", encoding="utf-8-sig") as f:
                    self.splited = json.load(f)
                    if (len(self.splited)) != len(self.sentences):
                        switch = True    
            else:
                switch = True
            if switch :
                self.splited, self.meanpooled_embeds = self.split2Clause(self.sentences)
                with open(self.filenames.splited_json, "w", encoding="utf-8-sig") as f:
                    json.dump(self.splited, f, ensure_ascii=False, indent=2)
                self.set_db()
            
            if not os.path.exists(self.filenames.embedding_np):
                self.embeds = self.clause_embedding(self.splited, print_highlighted= False)
            else:
                self.embeds = None

    def make_nd(self, obj: list, target_depth: int = 2):
        """
        리스트의 현재 중첩 깊이를 계산하고, target_depth만큼 감싸서 맞춤.

        Args:
            obj (list): 입력 리스트
            target_depth (int): 목표 차원 깊이 (예: 2 → 2D, 3 → 3D)

        Returns:
            tuple: (수정된 리스트, 원래 깊이)
        """
        depth = 0
        _obj = obj
        while isinstance(_obj, list):
            if not _obj:
                break
            _obj = _obj[0]
            depth += 1

        if depth > target_depth:
            raise ValueError(f"Too much depth ({depth}) for target {target_depth}!")

        for _ in range(target_depth - depth):
            obj = [obj]

        return obj, depth

    def split2Clause(self, data):
        """
        Tagging 모델을 활용해 입력 문장을 절(clause) 단위로 분할하고, 각 절의 mean pooling 임베딩을 생성합니다.

        동작 방식:
        - 문장을 토크나이징하여 태깅 결과(E/E2/E3)를 기준으로 절을 나눕니다.
        - 절의 최소 길이(threshold) 이하인 경우 삭제합니다.
        - 각 절마다 hidden state에서 mean pooling된 임베딩 벡터를 추출합니다.

        Args:
            data (List[str] or List[List[str]]): 분리할 문장 리스트 또는 문장 그룹

        Returns:
            Tuple[
                List[List[List[str]]],      # 분리된 절 리스트 → [video][sentence][clause]
                List[List[List[Tensor]]]   # 절별 mean pooling 임베딩 벡터 → [video][sentence][clause][768]
            ]
        """

        total, depth = self.make_nd(data, target_depth=2)
        
        results, embedding_t = [], []  # 최종 결과 (절 단위 문장 리스트)와 임베딩 벡터 리스트
        print("split to clause : ")
        for video in tqdm(total):  # 각 'video' 단위로 문장 그룹 처리 (예: 하나의 문서나 샘플)
            video_sents, embedding_v = [], []  # 해당 video에 대한 문장 및 임베딩 결과
            cls_temp = []
            for sentence in video:
                # 문장을 모델에 넣어 예측 수행
                # return_cls=True: [CLS] 벡터 포함, return_lhs=True: hidden state 반환
                predicted = prediction(
                    self.model, self.tokenizer, sentence,
                    LabelData().id2label, return_cls=True, return_lhs=True
                )

                cls_temp.append(predicted[1])  # [CLS] 벡터 저장
                embeddings = predicted[2][1:-1]        # [seq_len, hidden_size] 형태의 임베딩
                predicted = predicted[0]               # 토큰 라벨링 결과 [(tok, label, confidence), ...]

                if len(embeddings) != len(predicted):
                    raise ValueError(f"length dismatch :{embeddings}, {len(predicted)}")

                # 구문들, 임시구문, 구문분리 index, trigger
                clauses, clause, clause_end_idx, switch = [], [], [], False

                for i, (tok, label, confidence) in enumerate(predicted):
                    # 특정 라벨(`E`, `E2`, ...)이고 신뢰도 임계값 초과하며 의미역이 아닌 경우, 절 분리 시작
                    if label in self.elist and confidence > self.threshold and not self.is_segm(tok, predicted[i][0]):
                        switch = True
                    elif switch:
                        # clause 경계에서 구절 복원
                        recovered = recover_wordpieces(clause)
                        # 짧은 구절은 앞 구절에 병합
                        if len(recovered.split()) < 2 and clauses:
                            clauses[-1] += ' ' + recovered.strip('. ')
                            clause_end_idx[-1] = i
                        else:
                            clauses.append(recovered)
                            clause_end_idx.append(i)
                        clause, switch = [], False  # 구절 초기화
                    clause.append(tok)  # 토큰 누적
                # 마지막 구절도 처리
                if clause:
                    clauses.append(recover_wordpieces(clause).strip('. '))
                    clause_end_idx.append(i + 1)

                while clauses : # 절 길이 제한
                    c= clauses[0]
                    if len(c.split()) <= self.config.clause_len_threshold:
                        # print(f"Warning: Short clause detected: {c}")
                        clauses.remove(c)
                        continue
                    c= clauses[-1]
                    if len(c.split()) <= self.config.clause_len_threshold:
                        # print(f"Warning: Short clause detected: {c}")
                        clauses.remove(c)
                        continue
                    break

                if clauses:
                    video_sents.append(clauses)
                else:
                    video_sents.append([''])

                # 해당 문장의 구절별 mean pooling 벡터 계산
                start, embeds = 0, []
                for i in clause_end_idx:
                    embeds.append(embeddings[start:i].mean(dim=0))  # [768]
                    start = i
                embedding_v.append(embeds)
            self.cls_vectors.append(cls_temp)
            results.append(video_sents)
            embedding_t.append(embedding_v)

        # 입력 depth에 따라 결과 형태 조정
        if depth == 1:
            return results[0], embedding_t[0]
        elif depth == 0:
            return results[0][0], embedding_t[0][0]
        return results, embedding_t

    def clause_embedding(self, splited, print_highlighted: bool = True, highlight: bool = True, sbert_option: bool = False):
        """
        분리된 절 리스트(splited)를 바탕으로:
        - DeBERTa 임베딩을 통해 각 절의 [CLS] 임베딩 추출
        - 각 토큰 벡터와 [CLS], tagging-model CLS, mean pooled 벡터 간 cosine similarity를 계산하여 의미 강조 단어 추출
        - embedding 및 강조 결과를 JSONL과 npy로 저장

        Args:
            splited (List[List[List[str]]]): 절 단위로 분리된 텍스트 [[video][sentence][clause]]
            print_highlighted (bool): 강조 단어 출력 여부 (콘솔 디버깅용)

        Returns:
            List[List[List[np.ndarray]]]: 절 단위 [CLS] 임베딩 벡터 리스트 (저장 여부에 따라 None 가능)

        Side effects:
            - ./saved_temp 폴더에 batch별 .npy 저장
            - significant_jsonl 파일에 clause + 강조 단어 JSONL 저장
            - embedding_np 파일로 모든 임베딩 병합 저장
        """
        
        def delete_history_log(save_dir: str, file_type = None):
            """
            save_dir 내에 존재하는 {file_type}_batch_*.npy 파일들을 찾아 삭제합니다.

            Args:
                save_dir (str): 저장된 배치 파일들이 있는 디렉터리
                output_path (str): 최종 출력 결과 경로 (해당 파일도 삭제함)
                file_type (str): 파일 접두어 (예: 'embedding', 'relation' 등)
            """
            if file_type is None:
                print("⚠️ file_type을 지정하세요.")
                return

            files = [
                f for f in os.listdir(save_dir)
                if f.startswith(f"{file_type}_batch_") and f.endswith(".npy")]
            for file in files:
                full_path = os.path.join(save_dir, file)
                if os.path.exists(full_path):
                    os.remove(full_path)
            print(f"{len(files)}개의 임시 파일 삭제됨.")


        def save_batch_npy(batch_result, save_dir, file, batch_idx):
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'{file}_batch_{batch_idx}.npy')
            np.save(path, np.array(batch_result, dtype=object))  # 읽을때 allow_pickle=True 필요


        with open(self.filenames.significant_jsonl, "w", encoding="utf-8") as f:
            pass  # 초기화
        self.embedding_model = self.embedding_model.to("cuda")
        all_result = [] if len(splited) < self.config.return_embed_max else None

        print("splited.shape \t cls_vectors.shape    mp_embeds.shape")
        print(get_shape(splited), get_shape(self.cls_vectors), get_shape(self.meanpooled_embeds))
        delete_history_log(self.filenames.saved_temp_dir, file_type="embedding")
        delete_history_log(self.filenames.saved_temp_dir, file_type="sbert")
        # batch iter [3707,44,2,768] -> [37,100,44,2,768]
        for batch_idx in range(0, len(splited), self.config.save_batch):
            start = batch_idx
            end = batch_idx + self.config.save_batch
            batch = splited[start:end]
            batch_cls_vectors = self.cls_vectors[start:end] if len(self.cls_vectors) else None
            if not self.meanpooled_embeds == None:
                batch_meanpooled_embeds = self.meanpooled_embeds[start:end] 

            result, sbert, highlighted = [], [], [[],[],[]]
            # video iter : for [44,2,768] in [100,44,2,768]
            for V in tqdm(range(len(batch)), desc=f"Batch {batch_idx // self.config.save_batch}"):
                temp_embed, temp_sbert, highlight_temp = [], [], [[],[],[]] 
                # sentence iter : for [2,768] in [44,2,768]
                for S in range(len(batch[V])): 
                    temp_video, _temp_sbert, highlight_video = [], [], [[],[],[]]
                    # clause iter : for [768] in [2,768]
                    for C in range(len(batch[V][S])):
                        s = batch[V][S][C]
                        if batch_cls_vectors:
                            s_cls = batch_cls_vectors[V][S].to('cuda')
                        if self.meanpooled_embeds != None:
                            s_mp_emb = batch_meanpooled_embeds[V][S][C].to('cuda')

                        # 구문 단위 [CLS] 임베딩 추출
                        inputs = self.tokenizer(text=s, return_tensors='pt', add_special_tokens=True)
                        input_ids = inputs["input_ids"]
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = self.embedding_model(**inputs)
                        hidden_states, cls_clause = outputs.last_hidden_state, outputs.last_hidden_state[:, 0, :]
                        temp_video.append(cls_clause.squeeze(0).cpu().numpy())
                        if sbert_option:
                            _temp_sbert.append(self.sbert(cls_clause.squeeze(0).cpu(), hidden_states[0]).detach().cpu().numpy())

                        if highlight:
                            # 단어 수준 의미 추출을 위한 사전처리
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

                            # 각 단어별 토큰 인덱스 집계
                            word2indices = {}
                            for word, idx in token_map:
                                word2indices.setdefault(word, []).append(idx)

                            # 각 단어의 벡터들과 [CLS] 벡터 간 cosine similarity 평균 계산
                            standard1 = cls_clause[0].unsqueeze(0)
                            standard2 = s_cls.unsqueeze(0) if batch_cls_vectors else None
                            standard3 = s_mp_emb.unsqueeze(0) if self.meanpooled_embeds else None

                            def similarity(standard):
                                if standard == None:
                                    return
                                word_scores = []
                                for word, indices in word2indices.items():
                                    vecs = torch.stack([hidden_states[0, i] for i in indices])

                                    sims = F.cosine_similarity(vecs, standard, dim=1)
                                    score = self.rms(sims)  
                                    word_scores.append((word, float(score)))
                                word_scores_sorted = sorted(word_scores, key=lambda x: x[1], reverse=True)                 

                                # 유사도 기준 상위 단어를 의미 강조 단어로 선정 (default 60%)
                                top_n = max(1, int(len(word_scores_sorted) * self.config.important_words_ratio))
                                top_words = {word for word, _ in word_scores_sorted[:top_n]}
                                return [word for word in real if word in top_words]

                            highlight_video[0].append(similarity(standard1))
                            highlight_video[1].append(similarity(standard2))
                            highlight_video[2].append(similarity(standard3))
                    
                    temp_embed.append(temp_video)
                    temp_sbert.append(_temp_sbert)
                    if highlight:
                        highlight_temp[0].append(highlight_video[0])
                        highlight_temp[1].append(highlight_video[1])
                        highlight_temp[2].append(highlight_video[2])
                result.append(temp_embed)
                sbert.append(temp_sbert)
                if highlight:
                    highlighted[0].append(highlight_temp[0])
                    highlighted[1].append(highlight_temp[1])
                    highlighted[2].append(highlight_temp[2])

            if print_highlighted and highlight:
                for sentences, h1,h2,h3 in zip(batch, highlighted[0],highlighted[1],highlighted[2]):
                    for a,b,c in zip(highlight(sentences, h1, True),highlight(sentences, h2, True),highlight(sentences, h3, True)):
                        print('C:',a)
                        print('S:',b)
                        if self.meanpooled_embeds != None:
                            print('E:',c)
                        print()
                    break
            if highlight:
                for clauses, highlights in zip(batch, highlighted[0]):
                    item = {"clause": clauses, "highlight": highlights}
                    with open(self.filenames.significant_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

            save_batch_npy(result, self.filenames.saved_temp_dir, file="embedding", batch_idx=batch_idx)
            if sbert:
                save_batch_npy(sbert, self.filenames.saved_temp_dir, file="sbert", batch_idx=batch_idx)
            if all_result is not None:
                all_result.extend(result)

        def load_and_merge_npy(save_dir: str, output_path: str,file_type = None):
            files = [f for f in os.listdir(save_dir) if f.startswith(f"{file_type}_batch_") and f.endswith(".npy")]
            if not files:
                raise FileNotFoundError("병합할 .npy 파일이 없습니다.")
            files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(files) == 1:
                src = os.path.join(save_dir, files[0])
                np.save(output_path, np.load(src, allow_pickle=True))
                return
            merged = []
            for file in files:
                batch_path = os.path.join(save_dir, file)
                data = np.load(batch_path, allow_pickle=True)
                merged.extend(data)
            np.save(output_path, np.array(merged, dtype=object))
            print(f"npy file merged! path: {output_path} and length is : ",len(merged))

        load_and_merge_npy(self.filenames.saved_temp_dir, self.filenames.embedding_np, file_type="embedding")
        if sbert:
            load_and_merge_npy(self.filenames.saved_temp_dir, self.filenames.sbert_np, file_type="sbert")
        return all_result

    def sbert(self, cls_clause, hidden_states, mode= 'mean'):
        projected = self.concat_project(cls_clause, hidden_states, mode=mode)
        return projected

    def is_gram(self, word):
        """주어진 단어가 조사(J), 어미(E), 접미사(XS)인지 여부 확인"""
        t = self.kiwi.tokenize(word)[-1].tag
        return t[0] in ['J', 'E'] or t[:2] == 'XS'

    def is_segm(self, word, prev):
        """두 토큰 결합 시 의미 단위(N/V/M/XR 등)로 분리될 수 있는지 판단"""
        combined = prev + word.strip('#') if word.startswith('#') else prev + ' ' + word
        t = self.kiwi.tokenize(combined)[-1].tag
        return t[0] in ['N', 'V', 'M'] or t[:2] == 'XR'

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        """Root Mean Square 연산 (유사도 평균 계산 시 활용)"""
        return torch.sqrt(torch.mean(x ** 2))

    def str2real(self, text, timecat=True, output_str=True):
        """
        텍스트를 형태소 분석하여 의미 요소만 추출
        timecat=True일 경우 시간 관련 숫자 묶음도 처리함
        """
        tokens = self.kiwi.tokenize(text)
        return ' '.join(self.bereal(tokens, timecat)) if output_str else self.bereal(tokens, timecat)

    def bereal(self, tokens, timecat=True):
        """
        형태소 분석 결과에서 의미 있는 형태소만 추출합니다.

        필터링 방식:
        - 주요 품사(tag): 명사, 동사, 숫자, 외래어 등 take 리스트에 해당하는 경우만 유지
        - 시간 관련 숫자 묶음(timecat=True): '2023년 1월' 등은 하나의 문자열로 결합 처리

        Args:
            tokens (List[Token]): kiwi.tokenize() 결과
            timecat (bool): 시간 묶음 처리 여부

        Returns:
            List[str]: 의미 기반 단어 시퀀스
        """
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

    def extract_tail_morphemes(self, eojeol: str) -> str:
        tokens = self.kiwi.tokenize(eojeol)
        tail_tags = {'EC', 'EF', 'ETN', 'ETM',  # 어미
                    'JKS', 'JKC', 'JKG', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',  # 조사
                    'XSN', 'XSV', 'XSA'}  # 접미사
        collected = []

        for tok in reversed(tokens):
            if tok.tag in tail_tags:
                collected.insert(0, tok)
            else:
                break  # 연속된 어미/조사/접미사가 끝나면 중단
        return self.kiwi.join(collected) if collected else ''

    def summary(self, sample: int = 0):
        """
        분리된 절과 해당 강조 단어, [CLS] 임베딩 벡터 정보를 요약 출력하는 함수
        - 중요 단어 강조 여부, 임베딩 개수, 문장/절 수 등 통계 제공
        Args:
            max_sentence: 출력할 문장 수 제한 (기본값 5)
        """
        try:
            # 만약 메모리에 없고, embedding_np 파일이 있다면 로드 시도
            if not self.embeds:
                if os.path.exists(self.filenames.embedding_np):
                    self.embeds = np.load(self.filenames.embedding_np, allow_pickle=True)
                    print("[INFO] 임베딩 로딩 완료 → shape:", get_shape(self.embeds))
                else:
                    print("[경고] embedding_np 파일이 존재하지 않아 출력할 수 없습니다.")
                    return
        except Exception as e:
            print("[에러] 임베딩 복원 실패:", e)
            return

        if sample > 0:
            print("\n📌 절 분리 및 강조 단어 예시:")
            for a in highlight_jsonl(self.filenames.significant_jsonl, sample=sample):
                print(a)
                print()

        # 절, 문장 수 관련 통계
        total_sentences = sum(len(sentence) for sentence in self.splited)
        total_videos = len(self.splited)
        avg_sentences = total_sentences / total_videos if total_videos > 0 else 0
        avg_clauses = self.history['num_clauses'] / total_sentences if total_sentences > 0 else 0


        print("\n📊 [분석 요약]")
        print(f"✅ 전체 비디오 수          : {total_videos}")
        print(f"✅ 전체 문장 수            : {total_sentences}")
        print(f"✅ 전체 구절 수            : {self.history['num_clauses']}")
        print(f"✅ 비디오당 평균 문장 수   : {avg_sentences:.2f}")
        print(f"✅ 문장당 평균 구절 수     : {avg_clauses:.2f}")
        print(f"✅ 추출된 관계 (triplets)  : {self.history['num_triplets']}개")
        print("\n🧾 [중요 변수 구조]")
        print(f"📌 self.splited          : {get_shape(self.splited)} ({type(self.splited).__name__})")
        print(f"📌 self.meanpooled_embeds: {get_shape(self.meanpooled_embeds)}")
        print(f"📌 self.embeds           : {get_shape(self.embeds)}")
        print(f"📌 self.cls_vectors      : {get_shape(self.cls_vectors)}")
        print(f"📌 self.sentences        : {get_shape(self.sentences)}")

    def find_rel(self):
        """
        형태소 분석기(kiwi)를 사용하여 분리된 절 간의 의미 관계(triplet)를 추출합니다.

        단서 유형:
        - 어미(aumi): 절 마지막에서 찾음
        - 접속사(conj): 절 시작에서 찾음
        - 어간(augan): 현재 미사용

        관계 처리:
        - 관계가 겹치는 경우 max(rel_id)를 적용
        - 단서/역할/관계 분류는 CSV에서 불러오며, '없음'은 기본값 0으로 매핑됩니다.

        Side effects:
            - self.history['num_triplets']에 관계 수 저장
            - triplets_np 경로에 (clause_id1, clause_id2, relation_id) 저장
        """
        relation_trigger_ = pd.read_csv(self.filenames.relation_trigger)
        relation_trigger = relation_trigger_[['단서', '역할', '최종분류']] # 단서, 역할, 관계분류
        relation_set = set([s.strip() for s in set(relation_trigger['최종분류'].unique())])
        aumi, conj, augan = [], [], []
        for _, row in relation_trigger.iterrows():
            pair = (row['단서'].strip(), row['최종분류'].strip())
            if row['역할'] == '어미':
                aumi.append(pair)
            elif row['역할'] == '접속사':
                conj.append(pair)
            elif row['역할'] == '어간':
                augan.append(pair)
            else:
                raise ValueError('We\'ve got wrong data')

        def aumi_exception(v, t_last): # True면 걸러짐
            term = v[0].strip('~')
            if term == '니까':
                for t in t_last:
                    if term in t.form:
                        if t.tag != 'EC':
                            return True
                        break
            elif term in ['하여','해']: # '하여', '해' 처리 
                for t in t_last:
                    if t.form ==  '하':
                        if t.tag != 'XSV':
                            return True
                    elif t.form == '어':
                        if t.tag != 'EC':
                            return True
            elif term == '니':
                for t in t_last:
                    if t.form == '니':
                        if t.tag != 'EC':
                            return True
            return False

        rel_map = {'없음': 0}
        rel_map.update({label: i+1 for i, label in enumerate(sorted(relation_set))})
        # rel_map = {'없음': 0, '기타': 1, '대조/병렬': 2, '상황': 3, '수단': 4, '역인과': 5, '예시': 6, '인과': 7}
        self.rel_map = rel_map  # save relation map for later use

        sentences = self.splited_id_mapping()
        print(f"관계 추출 시작: {get_shape(sentences)}개의 문장, {len(self.splited)}개의 비디오에서 관계 추출")
        triplets = []    # video 단위로 안 나눔.
        print("Finding relations...")
        for sentence in tqdm(sentences):
            triplet_temp = {}
            for c_idx in range(len(sentence)):
                clause = sentence[c_idx][1] 
                rel = rel_map['없음'] # initialize
                # aumi
                last = ' '.join(clause.split(', ')[-2:]).strip('. ') # 마지막 두 어절
                t_last = self.kiwi.tokenize(last)
                for v in aumi:
                    if any(v[0].strip('~ ') in t.form for t in t_last):
                        if aumi_exception(v, t_last):
                            continue
                        rel = rel_map[v[1]]
                        break
                if c_idx < len(sentence) - 1:
                    triplet_temp[(sentence[c_idx][0], sentence[c_idx+1][0])] = rel # id만 저장
                # conj
                first = ' '.join(clause.split()[:2]).strip('. ') # 첫 두 어절
                t_first = self.kiwi.tokenize(first)
                for v in conj:
                    if any(v[0].strip() in t.form for t in t_first):
                        rel = rel_map[v[1]]
                        break
                if c_idx > 0:
                    triplet_temp[(sentence[c_idx-1][0], sentence[c_idx][0])] = max(rel, triplet_temp.get((sentence[c_idx-1][0], sentence[c_idx][0]), 0))
            triplets.extend([(id1, id2, rel) for (id1, id2), rel in triplet_temp.items()])
        # save triplets
        triplets_np = np.array(triplets, dtype=np.int32)  # shape: [N, 3] (id, id, rel)
        np.save(self.filenames.triplets_np, triplets_np)
        self.history['num_triplets'] = len(triplets)
        print(f"관계 추출 완료: {self.history['num_triplets']}개의 관계 추출됨.")

    def set_db(self):
        """
        분리된 절(self.splited)을 clause_id 기반으로 DB에 저장합니다.

        - clause_id = V*100000 + S*10 + C 형식으로 생성
        - 너무 긴 S(문장 인덱스 > 9999)나 C(절 인덱스 > 9)는 제외
        - 일정 개수마다 배치로 insert하여 성능 향상

        Side effects:
            - clause_data 테이블에 절 내용 저장
            - self.history['num_clauses']에 절 수 저장
            - ClauseDB.close() 호출하여 DB 종료
        """
        db = ClauseDB(self.filenames.clause_db, self.filenames.embedding_np)
        batch = [] # batchsize: 1000개
        for V, video in enumerate(self.splited): # max 없음
            for S, sentence in enumerate(video): # max 10000
                if S >= 10000:
                    print("Eliminated VS:",V,S)
                    break # 오버하면 버림
                for C, clause in enumerate(sentence): # max 10
                    if not isinstance(clause,str):
                        raise ValueError(f"Clause has a problem! : {clause}")
                    if C >= 10:
                        print("Eliminated VSC :",V,S,C)
                        break # 오버하면 버림
                    clause_id = V*100000 + S*10 + C
                    batch.append((clause_id, clause))

                    if len(batch) >= 1000:  # batch insert
                        db.insert_batch(batch)
                        batch = []
        if batch:
            db.insert_batch(batch)
        self.history['num_clauses'] = db.count_clauses()
        db.close()

    def splited_id_mapping(self) -> list:
        """
        db: ClauseDB instance
        """
        db = ClauseDB(self.filenames.clause_db, self.filenames.embedding_np)
        result = db.get_all_clauses(return_format='sents', return_id=True)

        if len(result) != self.history['num_clauses']:
            print(f"Warning: Mismatch in clause count! Expected {self.history['num_clauses']}, got {len(result)}")
        db.close()
        return result

    def print_triplets(self, number = float('inf'), triplets: np.ndarray = None):
        """
        DB에 저장된 절(clause)을 바탕으로, 저장된 관계(triplets)를 사람이 읽을 수 있는 형태로 출력합니다.

        색상:
        - 없음(0): 파랑
        - 주요 관계(1~6): 노랑
        - 특수관계(7): 마젠타

        Args:
            number (int): 출력할 최대 관계 수
            triplets (np.ndarray, optional): 미리 불러온 (id1, id2, rel_id) 배열. 없으면 파일에서 자동 로드
        """
        db = ClauseDB(self.filenames.clause_db, self.filenames.embedding_np, self.rel_map)
        if triplets is None:
            if not os.path.exists(self.filenames.triplets_np):
                print("No triplets found. Please run find_rel() first.")
                return
            triplets = np.load(self.filenames.triplets_np)
        i=0
        for id1, id2, rel_id in triplets:
            clause1 = db.get_clause(id1)
            clause2 = db.get_clause(id2)
            if rel_id == 7:
                color = ('\033[95m', '\033[0m')
            elif rel_id == 0:
                color = ('\033[94m', '\033[0m')
            else:
                color = ('\033[93m', '\033[0m')
            rel = db.rev_map.get(rel_id, '없음')
            if clause1 and clause2:
                print(f"{clause1}  -({color[0]}{rel}{color[1]})->  {clause2}")
            else:
                print(f"Invalid triplet: ({id1}:{clause1}, {id2}:{clause2}, {rel_id}:{rel})")
            if i >= number:
                break
            i += 1
        db.close()

class ClauseDB:
    """
    절(clause) 단위 데이터를 저장하고 관리하기 위한 SQLite 기반 데이터베이스 클래스.

    주요 역할:
    - 각 절을 고유 ID(clause_id)로 저장하고 조회 가능하게 구성
    - clause_id = V*100000 + S*10 + C 로 구성되어 위치 정보를 압축 표현
    - numpy 기반 임베딩 벡터도 함께 관리 (메모리 + 파일 저장 가능)
    - 관계 ID 매핑(rel_map)도 함께 보관하여 triplet 출력에 활용

    저장 구조:
    - 절 텍스트: SQLite의 clause_data 테이블 (id, clause)
    - 절 임베딩: numpy 파일(allow_pickle=True)로 [V][S][C][768] 구조 저장

    Args:
        db_path (str): SQLite DB 파일 경로
        embedding_path (str): 임베딩 저장/불러오기용 npy 파일 경로
        rel_map (dict): 관계 ID와 명칭 간 매핑 정보 (ex: {'없음': 0, '원인': 1, ...})
    """
    
    def __init__(self, db_path, embedding_path, rel_map: dict = None):
        print("[ClauseDB] Opening DB:", db_path)
        self.rel_map = rel_map if rel_map else {'없음': 0, '기타': 1, '대조/병렬': 2, '상황': 3, '수단': 4, '역인과': 5, '예시': 6, '인과': 7}
        self.rev_map = {v: k for k, v in rel_map.items()} if rel_map else {0: '없음'}
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        # self.cur.execute("PRAGMA journal_mode = WAL") # 속도 향상
        # self.cur.execute("PRAGMA synchronous = OFF") # 속도 향상 but 손실 가능
        self.embedding_path = embedding_path
        self._create_table()

        if os.path.exists(self.embedding_path):
            self.embeddings = list(np.load(self.embedding_path, allow_pickle=True))  # list of video
        else:
            self.embeddings = []

    def id2VSC(self, clause_id):
        clause_id = int(clause_id)
        V = clause_id // 100000
        S = (clause_id % 100000) // 10
        C = clause_id % 10
        return V,S,C

    def f5(self):
        if os.path.exists(self.embedding_path):
            self.embeddings = list(np.load(self.embedding_path, allow_pickle=True))
            return True
        else:
            return False

    def _create_table(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS clause_data (
                id INTEGER PRIMARY KEY,
                clause TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def insert_batch(self, batch: list[tuple[int, str]]):
        """
        절 텍스트를 id와 함께 SQLite DB에 일괄 삽입합니다.

        Args:
            batch (List[Tuple[clause_id, clause]]): 절 ID와 텍스트 쌍 리스트
        """
        if not batch:
            print("⚠️ [insert_batch] 비어있는 배치가 들어왔습니다. 처리하지 않습니다.")
            return

        valid_batch = []
        for i, item in enumerate(batch):
            try:
                int(item[0])
            except ValueError:
                print(f"❌ [insert_batch] 잘못된 형식 @ index {i}: {item}")
            if (not isinstance(item, tuple) or
                len(item) != 2 or
                not isinstance(item[1], str)):
                print(f"❌ [insert_batch] 잘못된 형식 @ index {i}: {item}")
                continue
            valid_batch.append(item)

        if not valid_batch:
            print("❌ [insert_batch] 유효한 항목이 없습니다. 삽입 생략.")
            return

        try:
            self.cur.executemany(
                "INSERT OR REPLACE INTO clause_data (id, clause) VALUES (?, ?)",
                valid_batch
            )
            self.conn.commit()
            # print(f"✅ [insert_batch] 총 {len(valid_batch)}개 삽입 완료.")
        except Exception as e:
            print(f"🔥 [insert_batch] DB 삽입 중 예외 발생: {e}")

    def insert_video(self, video_embeddings: list[list], clause_items: list[tuple[int, str]], auto_save=True):  
        """
        하나의 video에 대한 clause 전체를 임베딩과 함께 DB에 삽입합니다.

        Args:
            video_embeddings (List[List[np.ndarray]]): 절 임베딩 [S][C][768]
            clause_items (List[Tuple[int, str]]): clause_id와 절 텍스트 쌍
            auto_save (bool): True일 경우 삽입 후 npy로 임베딩 저장 수행
        """
        print("check embedding shape:", get_shape(self.embeddings), end=' -> ')
        self.insert_batch(clause_items)
        self.embeddings.append(np.array(video_embeddings))  # shape: [S][C][768]
        print(get_shape(self.embeddings))
        if auto_save:
            self.save_embedding_files()
        else:
            print("You have to execute \"save_embedding_files()\" function to save!!")

    def save_embedding_files(self):
        """
        현재까지 메모리에 쌓인 모든 임베딩 리스트(self.embeddings)를 numpy 파일로 저장합니다.
        저장 시 ragged 배열이므로 dtype=object와 allow_pickle=True를 사용합니다.
        """
        np.save(self.embedding_path, np.array(self.embeddings, dtype=object))  # ragged OK

    def get_clause(self, clause_id):
        """
        [clause_id] -> [clause] id에 맞는 절 텍스트를 DB에서 조회합니다.
        """
        clause_id = str(clause_id)
        self.cur.execute("SELECT clause FROM clause_data WHERE id=?", (clause_id,))
        row = self.cur.fetchone()
        return row[0] if row else None

    def get_embedding(self, clause_id):
        """
        [clause_id] -> [embedding vector] id에 맞는 임베딩벡터를 반환합니다.
        """
        self.f5()
        V,S,C = self.id2VSC(clause_id)
        if V < len(self.embeddings) and S < len(self.embeddings[V]) and C < len(self.embeddings[V][S]):
            return self.embeddings[V][S][C]
        return None

    def get_all_embedding(self, return_id = False, return_dict = False):
        self.f5()
        flatten = {} if return_dict else []
        print("Embedding number : ",len([clause for video_unit in self.embeddings for sentence in video_unit for clause in sentence]))
        for V, video in tqdm(enumerate(self.embeddings), desc="Embedding 로딩"):
            if video is None: continue
            for S, sentence in enumerate(video):
                if sentence is None: continue
                for C, emb in enumerate(sentence):
                    if emb is None: continue
                    clause_id = V * 100000 + S * 10 + C
                    if return_dict:
                        flatten[clause_id] = emb
                    else :
                        flatten.append((clause_id,emb) if return_id else emb)
        return flatten 
        
    def get_id(self, clause_text):
        """
        !!! 추천하지 않습니다. 주의: 절 텍스트는 고유해야 합니다. !!!
        [clause_text] -> [clause_id] 절 텍스트에 해당하는 id를 DB에서 조회합니다.

        Args:
            clause_text (str): 절 텍스트

        Returns:
            int 또는 None: 해당 텍스트에 매칭되는 id (없으면 None)
        """
        self.cur.execute("SELECT id FROM clause_data WHERE clause=?", (clause_text,))
        row = self.cur.fetchone()
        return int(row[0]) if row else None

    def get_all_clauses(self, return_format="videos", return_id=False):
        """
        DB에 저장된 모든 절을 다양한 형식으로 반환합니다.

        Args:
            return_format (str): 'videos', 'sents', 'clauses' 중 선택
                - videos: [V][S][C] 구조  / (id, clause)
                - sents: [V*S][C] 구조    / (id, clause)
                - clauses: clause 리스트  / {clause_id: clause} 딕셔너리
            return_id (bool): False일 경우 id 없이 절 텍스트만 반환

        Returns:
            List or Dict: 선택된 형식의 절 데이터
        """
        if return_format not in ('videos','sents','clauses'):
            raise ValueError(f"Invalid return_format: '{return_format}'. Choose from \[videos, sents, clauses\]")
        
        self.cur.execute("SELECT id, clause FROM clause_data")
        rows = self.cur.fetchall()
        print(f"get_all_clauses : Fetched {len(rows)} clauses from the database.")

        if return_format == "clauses":
            return dict(rows) if return_id else [clause for _, clause in rows]
        
        video = []

        for clause_id, clause in rows:
            V,S,C = self.id2VSC(clause_id)
            while len(video) <= V: # V 차원 확장
                video.append([])
            while len(video[V]) <= S: # S 차원 확장
                video[V].append([])
            while len(video[V][S]) <= C: # C 차원 확장
                video[V][S].append(None)

            video[V][S][C] = clause if not return_id else (clause_id, clause)
        if return_format == "videos":
            return video
        
        result = []
        for video_sents in video:
            for sentence in video_sents:
                result.append(sentence[:])
        return result
    
    def count_clauses(self):
        self.cur.execute("SELECT COUNT(*) FROM clause_data")
        length = self.cur.fetchone()[0]
        return length
    
    def close(self):
        self.conn.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            print(f"[ClauseDB] Error occurred: {exc_value}")
        else:
            print("[ClauseDB] Closed successfully.")

    def reset_database(self):
        """
        clause_data 테이블의 모든 내용을 삭제합니다.
        임베딩(npy 파일)은 유지됩니다.
        """
        self.cur.execute("DELETE FROM clause_data")
        self.conn.commit()
        print("[ClauseDB] clause_data 테이블이 초기화되었습니다.")

    def reset_embeddings(self):
        """
        메모리 내 임베딩 리스트와 저장된 .npy 파일을 초기화합니다.
        SQLite DB는 유지됩니다.
        """
        self.embeddings = []
        if os.path.exists(self.embedding_path):
            os.remove(self.embedding_path)
            print(f"[ClauseDB] 임베딩 파일 삭제됨: {self.embedding_path}")
        else:
            print("[ClauseDB] 삭제할 임베딩 파일이 없습니다.")

    def update_embedding(self, clause_id, embedding: np.ndarray):
        if len(self.embeddings) == 0:
            if not self.f5():
                raise FileNotFoundError("There's no embedding saved.")
        V,S,C = self.id2VSC(clause_id)
        try:
            self.embeddings[V][S][C] = embedding
        except IndexError as e:
            print('\033[95m'+"Embedding vector is not founded."+'\033[0m', e, f"[{V} {S} {C}]")
        self.save_embedding_files()
        self.f5()
    
    def delete_clause(self, clause_id: int):
        """
        clause_data 테이블에서 해당 ID의 텍스트를 삭제합니다.
        """
        clause_id = int(clause_id)
        self.cur.execute("DELETE FROM clause_data WHERE id=?", (clause_id,))
        self.conn.commit()

class ConcatProject(nn.Module):
    """
    SBERT 방식 문장 임베딩 생성기
    - [CLS] + mean/max pooled vector → concat → projection
    """
    def __init__(self, input_size = 768):
        super(ConcatProject, self).__init__()
        self.device = 'cuda'
        self.linear = nn.Linear(input_size * 2, input_size).to(self.device)

    def forward(self, cls_vector, hidden_states, mode='mean'):
        cls_vector = cls_vector.to(self.device)
        if mode == 'mean':
            pooled = hidden_states[1:-1].mean(dim=0).to(self.device)  # [H]
        elif mode == 'max':
            pooled = hidden_states[1:-1].max(dim=0).values.to(self.device)  # [H]
        else:
            raise ValueError(f"Unsupported pooling mode: {mode}")

        if pooled.shape != cls_vector.shape:
            raise ValueError(f"Shape mismatch: pooled {pooled.shape}, cls_vector {cls_vector.shape}")
        
        concatted = torch.cat([cls_vector, pooled], dim=0)  # [2H]
        projected = self.linear(concatted)  # [H]
        return projected


def main():
    config = Config()
    config.confidence_threshold = 0.15 # 구문 분리 기준, 높을 수록 많이 잘림림
    config.important_words_ratio = 0.5 # 중요 키워드 기준, 높을 수록 많이 탐지
    config.clause_len_threshold = 3    # 구문 길이 제한, 어절 단위 

    dir_ = "./top_6.json"
    file = "./top_6_parsed.json"
    filtered_file = "./top_6_filtered.json"
    if os.path.exists(file):
        print("preprocessed file detected!")
        with open(file, "r", encoding="utf-8-sig") as f:
            sentences = json.load(f)
    else:
        sentences = select_terms(open_and_preprocess(dir_,file),filtered_file)

    sentences = sentences
    cs = ClauseSpliting(sentences, e_option= 'E3', threshold= True)
    cs.find_rel()
    cs.print_triplets(40)
    cs.summary(0)
    
if __name__ == "__main__":
    main()
