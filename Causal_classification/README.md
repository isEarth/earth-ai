[Kakaobank KF-DeBERTa-Base](https://huggingface.co/kakaobank/kf-deberta-base) 모델을 인과관계 문장 분류 Task를 위해 파인튜닝함.<br>
<br><br>

# 1. data

| **youtube 스크립트 스크래핑 데이터**                                                                                                                        | **모두의 말뭉치 의미역분석 말뭉치**                                |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| 1. SBS biz news – 262건<br>2. 한국경제 tv – 756건<br>3. 연합뉴스 경제tv – 319건<br>4. 서울경제tv 뉴스 – 1,163건<br><br>3,707건 스크래핑 → kss 활용 문장 분리<br> → 총 112,206개 문장 | SRL 라벨 : ARGM-CAU<br> 이유/원인(cause)이 라벨링된 문장 추출<br> → 총 13,432개 문장 |

**Total: 125,638개 문장**

| label             | sentence |
|-------------------|----------|
| 0(인과관계 포함)     | 76,170   |
| 1(인과관계 미포함)    | 49,468   |

<br><br>

# 2. patterns.py
데이터 라벨링을 위한 정규식 패턴 정의  
`참고: 한국어 텍스트 단위의 비명시적 인과 접속 연구. 정주연; 고려대학교 대학원, 2022.`  
<br><br>

# 3. dataset.py 
```bash
python Dataset.py --src <raw_texts_dir/> --dst <train.csv>
python Dataset.py --src /home/eunhyea/EARTH/ConceptMap/topic/download_folder/ --dst train.csv
```
정규식 기반 데이터 라벨링 및 문장 분리 후 (sentence,label) 형식으로 csv 파일 생성  
<br><br>

# 4. train.py
```bash
python train.py
```


학습 및 best model 저장, 성능평가 지표 저장  
| ![image 1](https://github.com/user-attachments/assets/0adb3fba-585b-4932-92ee-6ba61dab98a6) | ![image 2](https://github.com/user-attachments/assets/17c87b4f-bafa-4fd0-9506-ecd11935c501) | ![image](https://github.com/user-attachments/assets/1afbb57e-1131-4939-a99f-aba043dd1e13)|
|---|---|---|

## 4.1. evaluation.py
학습 중단 시 마지막 checkpoint 찾아서 성능평가 지표 저장
<br><br>

# 5. cls_module.py
```python
python cls_module.py -test golden_causal.csv
python cls_module.py -test input_test.txt
```
csv 혹은 txt 파일을 읽어와서 binary 분류 추론
