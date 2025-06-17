# Clause-Level Semantic Similarity Pipeline

## 프로젝트 개요
**문장 또는 절(clause) 단위 임베딩**을 기반으로, **대규모 텍스트** 사이의 **의미 유사성**을 효율적으로 탐색하고, 구조화된 텍스트 관계를 추출하는 전체 파이프라인입니다.  

- **PyTorch 기반 학습 및 추론**
- **Custom 차원 축소/확장 모듈 빠른 유사도 검색**
- **절 경계 예측, 임베딩 생성, 유사 절쌍 탐색, 전처리까지 포함한 엔드투엔드 시스템**
- **절 당 중요단어 추출 및 강조 (선택)**


## 프로젝트 플로우
```

[Raw Text]
↓
[processing.py] 문장 전처리 (문장분리, 오타/구어체 보정, 경제용어 기반 filtering)
↓
[prediction.py] 문장을 절 단위로 분할 (KF-DeBERTa + finetuned custom TaggingModel)
↓
[embedding] 절 단위 벡터 임베딩 생성 및 s-bert기법 가공
↓
[decide_same.py] 의미 유사 절쌍 탐색 (cosine similarity + tf-idf + Kmean clustering)
↓
[test.py] 결과 시각화
```
<br>

<p align="center">

![alt text](images/image-2.png)
</p>
<p align="center">
<img src="images/image-3.png" width="500"/> 
</p>

---
## 주요 기술 스택

| Task               | Tool/Library                |
|--------------------|-----------------------------|
| Transformer 모델   | HuggingFace Transformers    |
| 임베딩            |   `KF-DeBERTa`           |
| Classifer         | custom `TaggingModel`   |
| 토큰화            | `DebertaV2Tokenizer`, `Kiwi`  |
| data 관리        | custom `ClauseDB`     |
| Parameter 관리    | `dataclass` 및 `config` |
| 유사도 계산       | `cosine_similarity`, `TF-TDF`, `KMean`  |
| 빠른 탐색         | 차원 축소 선탐색 후 정밀탐색  |
| 시각화/진행상황    | `tqdm`, `matplotlib`         |

---

## 실행 방법
1. 환경 설정
```
- python 3.9
- torch 2.5.1
- transformers 4.49.0
- tqdm 4.67.지
├── train.py               # 문장 분류기 학습
├── prediction.py          # 문장을 절 단위로 분리
├── decide_same.py         # 유사 절쌍 탐색 및 정밀 유사도 계산
├── test.py                # 전체 파이프라인 검증
└── requirements.txt       # 환경 설정
```
