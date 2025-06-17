"""
dataset.py

한글 텍스트 파일(.txt)들로 구성된 디렉토리에서 인과 문장을 자동 추출하여
CSV 학습 데이터셋을 생성합니다. 문장은 `kss`를 이용해 분리하고,
`kiwipiepy` 기반 토크나이저로 정규표현식 기반 인과 패턴을 탐지합니다.

사용 예시:
    python dataset.py --src raw_texts_dir/ --dst train.csv

출력:
    train.csv_YYYYMMDD_HHMMSS.csv 파일이 생성됩니다.
    컬럼: sentence, label (label은 0 또는 1)

요구 사항:
    - KSS (문장 분리용)
    - kiwipiepy (형태소 분석기)
    - patterns.py (CAUSAL_PATTERNS 정규식 목록 포함되어야 함)
"""

# dataset.py
# 사용 예시:
#   python dataset.py --src raw_texts_dir/ --dst train.csv
#   python dataset.py --src /home/eunhyea/EARTH/ConceptMap/topic/download_folder/ --dst train.csv

import re
import csv
from pathlib import Path
import datetime

from kiwipiepy import Kiwi
import kss                          # 한글 문장 분리를 위해 kss 사용
from tqdm import tqdm               # 진행 상황 표시용
from patterns import CAUSAL_PATTERNS  # 인과 패턴(정규식) 정의 파일

# -------------------------------------------------------------------
# 1) 인과 패턴 정규식을 한 번만 컴파일해 둡니다.
compiled_patterns = [re.compile(p) for p in CAUSAL_PATTERNS]

kiwi = Kiwi()

def has_causal_phrase(sentence: str) -> bool:
    """
    문장에 인과 관계 패턴(정규식)이 하나라도 있으면 True 반환
    """
    tok = kiwi.tokenize(sentence, compatible_jamo=True)
    tok_sen = ' '.join([i.form for i in tok])
    return any(p.search(tok_sen) for p in compiled_patterns)

# -------------------------------------------------------------------
def build_csv(src_dir: str, dst_csv: str):
    """
    src_dir: 원본 .txt 파일들이 모여 있는 디렉토리 경로
    dst_csv: 'sentence,label' 형식으로 저장할 CSV 파일 경로

    동작 순서:
      1) src_dir 내부의 모든 .txt 파일을 찾아서 내용을 읽어들임
      2) 각 파일의 텍스트를 하나의 큰 문자열로 합침
      3) kss.split_sentences(...)로 문장 단위 분리 → 리스트
      4) 각 문장마다 has_causal_phrase(...)로 라벨(0/1) 결정
      5) 결과를 CSV로 저장 (헤더: sentence,label)
    """
    src_path = Path(src_dir)
    if not src_path.exists() or not src_path.is_dir():
        print(f"오류: '{src_dir}' 디렉토리를 찾을 수 없거나 디렉토리가 아닙니다.")
        return

    # 2) 디렉토리 내부의 모든 .txt 파일 목록을 가져옴 (하위 디렉토리 미탐색)
    txt_files = list(src_path.glob("*.txt"))
    if len(txt_files) == 0:
        print(f"'{src_dir}' 디렉토리에 .txt 파일이 하나도 없습니다.")
        return

    # 3) 모든 파일 내용을 합쳐서 하나의 큰 문자열로 만들기
    all_text = []
    print("텍스트 파일 읽는 중...")
    for txt_file in tqdm(txt_files, desc="Reading .txt files", unit="file"):
        try:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            print(f"파일 읽기 실패: {txt_file} → {e}")
            continue

        if len(lines) > 1:
            content = "\n".join(lines[1:]).strip()  # 두 번째 줄부터 끝까지 합침
            all_text.append(content)
        else:
            print(f"'{txt_file.name}' 파일에 두 번째 줄이 없습니다. 건너뜁니다.")

    # 이제 all_text는 디렉토리 내 각 텍스트 파일의 내용을 담은 리스트
    # 이를 하나의 큰 문자열로 합칩니다.
    combined_text = "\n".join(all_text).strip()
    if not combined_text:
        print("주어진 디렉토리 내 모든 텍스트를 읽었으나, 내용이 비어있거나 모두 건너뛰었습니다.")
        return

    # 4) kss를 이용해 한글 문장 단위로 분리
    print("kss 문장 분리 중...")
    try:
        sentences = kss.split_sentences(combined_text, backend='mecab')
    except Exception as e:
        print("kss 문장 분리 중 오류 발생:", e)
        return

    # 5) CSV 파일 작성 (헤더: sentence,label)
    print("CSV 생성 중...")
    with open(dst_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label"])  # 헤더

        count = 0
        # tqdm을 사용하여 문장 처리 진행 상황을 표시
        for sent in tqdm(sentences, desc="Processing sentences", unit="sent"):
            sent = sent.strip()
            if not sent:
                continue

            # 인과 패턴 매칭 → 인과 문장이면 1, 아니면 0
            label = 1 if has_causal_phrase(sent) else 0
            writer.writerow([sent, label])
            count += 1

    print(f"'{dst_csv}' 생성 완료 - 총 {count}개 문장 처리됨")

# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="디렉토리 내 모든 .txt 파일을 읽어서 문장 단위 CSV 생성"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="원본 .txt 파일들이 들어있는 디렉토리 경로"
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="생성할 CSV 파일 경로 (sentence,label 형식)"
    )
    args = parser.parse_args()

    # 현재 날짜 및 시간 추가
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 최종 CSV 파일명 만들기
    output_csv = f"{args.dst}_{timestamp}.csv"

    build_csv(args.src, output_csv)
