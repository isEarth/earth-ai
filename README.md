# 지능정보 SW 6기 자연어처리 8조 - 지구다

<table>
 <tr>
    <td align="center"><a href="https://github.com/eunhyea"><img src="https://avatars.githubusercontent.com/eunhyea" width="150px;" alt=""></td>
    <td align="center"><a href="https://github.com/eunwookim"><img src="https://avatars.githubusercontent.com/eunwookim" width="150px;" alt=""></td>
    <td align="center"><a href="https://github.com/Auspiland"><img src="https://avatars.githubusercontent.com/Auspiland" width="150px;" alt=""></td>
    <td align="center"><a href="https://github.com/DoxB"><img src="https://avatars.githubusercontent.com/DoxB" width="150px;" alt=""></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/eunhyea"><b>고은혜</b></td>
    <td align="center"><a href="https://github.com/eunwookim"><b>김은우</b></td>
    <td align="center"><a href="https://github.com/Auspiland"><b>윤정한</b></td>
    <td align="center"><a href="https://github.com/DoxB"><b>임정규</b></td>
  </tr>
  <tr>
    <td align="center">Python</td>
    <td align="center">Python</td>
    <td align="center">Python</td>
    <td align="center">Python</td>
  </tr>
</table>

<br />
<br />

# 지구다 AI R&D Result

## 1. causal_classification
바로가기: [causal_classificaton](causal_classification/README.md)
## 프로젝트 구조
```
Causal_classification/
├── cls_module.py        # 인과 관계 binary 분류 추론
├── dataset.py           # 정규식 기반 데이터 라벨링 및 문장 분리 후 (sentence,label) 형식으로 csv 파일 생성
├── train.py             # 학습 및 best model 저장, 성능평가 지표 저장
├── evaluation.py        # 학습 중단 시 마지막 checkpoint 찾아서 성능평가 지표 저장
├── patterns.py          # 데이터 라벨링을 위한 정규식 패턴 정의
└── requirements.txt     # 환경 설정
```
<br />

## 2. clause_split
바로가기: [clause_split](clause_split/README.md)

<br />

## 3. R-VGAE
바로가기: [R-VGAE](rvge/README.md)
## 프로젝트 구조
```
RVGAE/
├── data/                # Input 데이터
├── model.py             # R-VGAE 모델 정의
├── predict.py           # 링크 및 타입 예측 실행
└── requirements.txt     # 환경 설정
```
<br />

## 4. topic_lda
바로가기: [topic_lda](topic_lda/README.md)
## 프로젝트 구조
```
Topic_LDA/
├── assets/                # 데이터
├── CustomTokenizer.py     # 명사 단위 토크나이저
├── lda_modeling.py        # 명사로만 이루어진 스크립트와 불용어 사전, 카운터 벡터를 전달받아 LDA 토픽 모델링 진행
├── main.py                # 전처리부터 LDA 토픽 모델링, 저장까지 관리
├── preprocess.py          # 스크립트 txt 파일들을 불러와 전처리
├── stopwords.py           # TF-IDF 분석 기반으로 불용어 사전 구축하여 불용어 처리
└── requirements.txt       # 환경 설정
```

## 5. ~appendix
바로가기: [Appendix](~appendix/README.md)
## 프로젝트 구조
```
~appendix/
├── dicdata.ipynb         # 용어사전 중복 제거를 위한 전처리 작업
└── conceptmap.ipynb      # 컨셉맵 구축을 위한 전처리 작업
```