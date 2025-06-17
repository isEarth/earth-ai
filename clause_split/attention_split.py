# KF-DeBERTa 기반 Attention 분석 및 명사구 추출 파이프라인
# ------------------------------------------------------------
# 이 파이프라인은 한국어 문장에 대해 KF-DeBERTa 모델의 attention을 분석하고,
# 명사구 단위로 의미 그룹을 추출하여 시각화합니다.
#
# 주요 기능:
# - KF-DeBERTa 모델의 attention 시각화
# - WordPiece → 어절 수준 token 재구성
# - 명사구 추출 및 attention 기반 의미 그룹화
# - 의미 흐름 파악을 위한 RMS 기반 score 계산 및 시각화

from dataclasses import dataclass
import matplotlib.pyplot as plt
from kobert_tokenizer import KoBERTTokenizer
import copy
from kiwipiepy import Kiwi
import torch
from transformers import AutoTokenizer, AutoModel

kiwi = Kiwi()

@dataclass
class Config:
    """
    모델 설정 및 attention 가중치 설정값을 정의하는 클래스

    Attributes:
        model (str): 사용할 huggingface 모델 이름
        higher_atten_weight (float): 명사구 병합 시 추가 attention 가중치
        comma_atten_weight (float): 콤마(,)가 있을 경우 attention 감소량
    """
    model : str = 'kakaobank/kf-deberta-base'
    higher_atten_weight : float = 0.1
    comma_atten_weight : float = -0.5

config = Config()

# 모델 및 토크나이저 불러오기
model = AutoModel.from_pretrained(config.model, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(config.model)

# 입력 텍스트 로딩
with open('./data/example_text.txt', 'r', encoding='utf-8') as f:
    raw = f.read()
economy = [row.strip() for row in raw.splitlines() if len(row) > 1]

# 배치 설정 및 인코딩
number_of_examples = 10
economy_inputs = economy[:number_of_examples]
inputs = tokenizer.batch_encode_plus(economy_inputs, padding=True, truncation=True, return_tensors='pt')

# 모델 추론
with torch.no_grad():
    out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# 마지막 레이어 attention 조정: 콤마 감쇠 적용
for i in range(number_of_examples):
    tokenized = tokenizer.tokenize(economy_inputs[i])
    for n, token in enumerate(tokenized):
        if token == ',':
            out.attentions[-1][i][:, n, n:] *= (1.0 + config.comma_atten_weight)

# rms(x, dim): 텐서 x의 dim 축 기준 root-mean-square 계산
def rms(x, dim=0):
    """
    Root Mean Square 계산 함수

    Args:
        x (Tensor): 입력 텐서
        dim (int): RMS 계산할 차원

    Returns:
        Tensor: RMS 결과
    """
    return torch.sqrt(torch.mean(x ** 2, dim=dim))

def is_noun_only(token: str, sentence: str) -> int:
    """
    주어진 단어가 명사로 구성되었는지 판별

    Args:
        token (str): 검사할 단어
        sentence (str): 전체 문장 (형태소 기반 위치 검출용)

    Returns:
        int: 1 (명사), 2 (명사+조사), 0 (아님)
    """
    toks = [t.form for t in kiwi.tokenize(token)]
    last_tok = kiwi.tokenize(token)[-1]
    morphs = kiwi.tokenize(sentence)
    forms = [m.form for m in morphs]
    tags = []
    for i in range(len(forms) - len(toks) + 1):
        if forms[i:i+len(toks)] == toks:
            tags = [morphs[i+j].tag for j in range(len(toks))]
            break
    if all(t in ['NNG','NNP','XR'] for t in tags) or (last_tok.form in ['ᆷ','음'] and last_tok.tag == 'EF'):
        return 1
    if len(toks) > 1 and all(t in ['NNG','NNP','XR'] for t in tags[:-1]) and tags[-1].startswith('J'):
        return 2
    return 0

def be_noun(token: str) -> str:
    """
    조사 제거 후 명사 부분만 반환

    Args:
        token (str): 입력 문자열

    Returns:
        str: 명사 부분만 포함된 문자열
    """
    tokens = kiwi.tokenize(token)
    for idx in reversed(range(len(tokens))):
        if tokens[idx].tag.startswith('J'):
            tokens.pop()
        else:
            break
    return kiwi.join(tokens)

def noun_combine(word_indices: list, sentence: str) -> tuple:
    """
    WordPiece 단위 토큰 인덱스를 명사구 기준으로 그룹핑

    Args:
        word_indices (List[List[int]]): WordPiece 인덱스 목록
        sentence (str): 원문 문장

    Returns:
        Tuple[List[List[int]], List[str]]: 병합된 인덱스, 병합된 명사구 문자열
    """
    noun_group, count, idx = [], 0, 0
    while idx < len(word_indices):
        ino = is_noun_only(sentence.split()[idx], sentence)
        if ino:
            count += 1
            if count > 1:
                word_indices[idx] = word_indices[idx-1] + word_indices[idx]
                word_indices.pop(idx-1)
                noun_group.append(idx-1)
            if ino == 2:
                count = 0
        else:
            count = 0
        idx += 1
    grouped_words, split_words, temp = [], [], []
    for i in range(len(sentence.split())):
        if i in noun_group:
            temp.append(i)
        elif temp:
            temp.append(i)
            split_words.append(temp)
            grouped_words.append(be_noun(' '.join([sentence.split()[j] for j in temp])))
            temp = []
        else:
            split_words.append([i])
    return split_words, grouped_words

def token2word_embed(sentence_embed: torch.Tensor, group_indices: list, additional_weight=0.0, method="rms") -> torch.Tensor:
    """
    Token-level attention을 word/phrase 단위로 압축

    Args:
        sentence_embed (Tensor): [T,T] 또는 [H,T,T] attention 텐서
        group_indices (List[List[int]]): 어절 또는 명사구 단위 인덱스
        additional_weight (float): 명사구에 부여할 추가 가중치
        method (str): 집계 방법: rms / mean / sum

    Returns:
        Tensor: [G,G] 형식 압축 attention 행렬
    """
    if sentence_embed.dim() == 3:
        sentence_embed = rms(sentence_embed, dim=0)
    G = len(group_indices)
    result = torch.zeros((G, G))
    for i, rows in enumerate(group_indices):
        for j, cols in enumerate(group_indices):
            submatrix = sentence_embed[torch.tensor(rows)][:, torch.tensor(cols)]
            if method == "mean":
                score = submatrix.mean()
            elif method == "rms":
                score = torch.sqrt(torch.mean(submatrix ** 2))
            elif method == "sum":
                score = submatrix.sum()
            else:
                raise ValueError("Unknown method")
            if len(cols) > 1:
                score *= (1.0 + additional_weight)
            result[i, j] = score
    return result

def aujeul(sentnum: int, noun_comb=False) -> tuple:
    """
    주어진 문장의 token attention → 어절 단위 attention 변환 + 명사구 병합

    Args:
        sentnum (int): 문장 인덱스
        noun_comb (bool): 명사구 병합 여부

    Returns:
        Tuple[Tensor, List[List[int]], Optional[List[str]]]: attention 행렬, 어절 인덱스, 명사구 목록
    """
    word_indices, current_word = [], []
    sentence = economy_inputs[sentnum]
    for idx, token in enumerate(tokenizer.tokenize(sentence)[:-1]):
        if token.startswith('▁'):
            if current_word:
                word_indices.append(current_word)
                current_word = []
            current_word = [idx]
        else:
            current_word.append(idx)
    if current_word:
        word_indices.append(current_word)
    outa = copy.deepcopy(out.attentions[-1][sentnum])[:-1][:-1]
    result = token2word_embed(outa, word_indices, method='rms')
    if noun_comb:
        groupedby_noun, grouped_words = noun_combine(word_indices, sentence)
        result = token2word_embed(result, groupedby_noun, config.higher_atten_weight, method="sum")
        return result, groupedby_noun, grouped_words
    return result, word_indices, None

def gradient(values: list) -> tuple:
    """
    Attention 벡터의 local minimum을 찾아 문장 경계 추정

    Args:
        values (list): attention score 리스트

    Returns:
        Tuple[List[int], List[float]]: local minimum 위치, 변화량
    """
    diff = [values[i+1] - values[i] for i in range(len(values)-1)]
    height = max(values) - min(values)
    threshold = height * 0.15
    local_min = [idx for idx in range(1, len(diff)) if diff[idx] > 0 and diff[idx-1] <= 0 and (abs(diff[idx]) >= threshold or abs(diff[idx-1]) >= threshold)]
    return local_min, diff

def cutting(values: list, indexing=True) -> list:
    """
    gradient 결과를 바탕으로 문장을 절 단위로 분리

    Args:
        values (list): attention 벡터
        indexing (bool): 인덱스 반환 여부

    Returns:
        List[List[int]]: 절 단위 인덱스 그룹
    """
    local_min, _ = gradient(values)
    cutted, start = [], 0
    for idx in local_min:
        idx += 1
        cutted.append(list(range(start, idx)) if indexing else values[start:idx])
        start = idx
    cutted.append(list(range(start, len(values)+1)) if indexing else values[start:])
    return cutted

def special_glue(grouped_sent: list, flat: list) -> tuple:
    """
    문장이 조사(J*)로 끝나면 다음 구와 병합

    Args:
        grouped_sent (List[str]): 구 분리된 문장
        flat (List[List[int]]): 인덱스 그룹

    Returns:
        Tuple[List[str], List[List[int]]]: 병합된 문장/인덱스 목록
    """
    i = 0
    while i < len(grouped_sent):
        kiwi_token = kiwi.tokenize(grouped_sent[i])
        if kiwi_token[-1].tag in ['JKS','JKC','JKG','JKO','JX'] and i < len(flat) - 1:
            flat[i] += flat[i+1]
            flat.pop(i+1)
            grouped_sent[i] += ' ' + grouped_sent[i+1]
            grouped_sent.pop(i+1)
        i += 1
    return grouped_sent, flat
