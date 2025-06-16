# KF-DeBERTa Fine Tuning: 인과관계 문장 분류기

## 프로젝트 개요
경제 및 금융 텍스트에서 **인과관계 문장을 자동 탐지**하기 위해, **도메인 특화 언어모델 KF-DeBERTa**를 파인튜닝하였습니다.<br>
본 모델은 DeBERTa-v2 아키텍처 기반으로, 범용 + 금융 말뭉치에 대해 사전학습된 모델입니다.

## 데이터 구성 및 생성
1. 데이터 출처:
YouTube 경제 뉴스 스크립트와 모두의 말뭉치(SRL 기반 의미역 분석)를 활용

2. 정규식 패턴 정의:
데이터 라벨링을 위한 정규식 패턴 정의
참고: 한국어 텍스트 단위의 비명시적 인과 접속 연구. 정주연; 고려대학교 대학원, 2022.

3. 데이터 라벨링 및 CSV 생성:
문장 분리 후, 정규식 기반 라벨링을 통해 (sentence, label) 형식의 CSV 파일 생성

## 파인튜닝 방법
- 사용 모델: [Kakaobank/kf-deberta-base](https://huggingface.co/kakaobank/kf-deberta-base)
- 파인튜닝 프레임워크: HuggingFace `Trainer`
- 입력 데이터: `train.csv` (`sentence`, `label` 컬럼)

- 학습 구성:
  - 학습/검증 분할: 80:20
  - 최대 epoch: 20
  - batch size: 16
  - learning rate: 2e-5
  - early stopping: patience 2
  - 평가 지표: accuracy, precision, recall, F1, ROC-AUC

- 학습 완료 시:
  - best model → `./runs/run_xxxx/best_model/`에 저장
  - 학습 곡선, 혼동행렬, ROC 곡선 자동 시각화 및 저장
## 성능 및 결과
| ![image 1](https://github.com/user-attachments/assets/0adb3fba-585b-4932-92ee-6ba61dab98a6) | ![image 2](https://github.com/user-attachments/assets/17c87b4f-bafa-4fd0-9506-ecd11935c501) | ![image](https://github.com/user-attachments/assets/1afbb57e-1131-4939-a99f-aba043dd1e13)|
|---|---|---|

## 실행 방법
1. 환경 설정
```
python 3.9.21
cuda 12.1
numpy 1.26.4
torch 2.5.1
transformers 4.49.0
kss 6.0.4
```
2. 의존성 설치
```
pip install -r requirements.txt
```
3. 실행
```
python cls_module.py --test input.csv  # or input.txt
```
## 데이터 구조
| **youtube 스크립트 스크래핑 데이터**                                                                                                                        | **모두의 말뭉치 의미역분석 말뭉치**                                |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| 1. SBS biz news – 262건<br>2. 한국경제 tv – 756건<br>3. 연합뉴스 경제tv – 319건<br>4. 서울경제tv 뉴스 – 1,163건<br><br>3,707건 스크래핑 → kss 활용 문장 분리<br> → 총 112,206개 문장 | SRL 라벨 : ARGM-CAU<br> 이유/원인(cause)이 라벨링된 문장 추출<br> → 총 13,432개 문장 |

**Total: 125,638개 문장**

| label             | sentence |
|-------------------|----------|
| 0(인과관계 포함)     | 76,170   |
| 1(인과관계 미포함)    | 49,468   |

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
