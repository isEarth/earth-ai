"""
dataset.py

텍스트 파일(.txt)들을 문장 단위로 분리하고, 정규표현식 기반 인과관계 여부를 자동 라벨링하여
CSV 형태로 변환해주는 데이터 전처리 스크립트입니다.

사용 예시:
    python dataset.py --src raw_texts_dir/ --dst train
    python dataset.py --src /path/to/texts/ --dst output/train_data

입력:
    - src 디렉토리 내부의 .txt 파일들 (각 파일은 2번째 줄부터 본문 시작)

출력:
    - `--dst_YYYYMMDD_HHMMSS.csv` 형식의 CSV 파일 생성
    - 각 문장은 `sentence,label` 형식으로 저장
"""

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
    주어진 문장 내에 인과 관계 패턴(정규식)이 존재하는지 판단합니다.

    Args:
        sentence (str): 검사할 문장 (한글)

    Returns:
        bool: 인과 관계 패턴이 하나라도 매칭되면 True, 없으면 False
    """
    tok = kiwi.tokenize(sentence, compatible_jamo=True)
    tok_sen = ' '.join([i.form for i in tok])
    return any(p.search(tok_sen) for p in compiled_patterns)

# -------------------------------------------------------------------
def build_csv(src_dir: str, dst_csv: str):
    """
    디렉토리 내 모든 .txt 파일에서 문장을 추출하고, 인과 여부 라벨링 후 CSV로 저장합니다.

    Args:
        src_dir (str): 입력 .txt 파일들이 위치한 디렉토리 경로
        dst_csv (str): 결과 CSV 파일 경로 (확장자는 자동으로 `.csv`)

    처리 과정:
        1. 모든 .txt 파일의 내용을 읽고 2번째 줄부터 텍스트 추출
        2. 문장 단위 분리 (KSS 사용)
        3. 인과관계 정규식 매칭하여 라벨링 (1: 포함, 0: 미포함)
        4. sentence,label 형식으로 CSV 저장

    Returns:
        None: 결과는 파일로 저장되며 함수 자체는 반환값이 없습니다.
    """
    src_path = Path(src_dir)
    if not src_path.exists() or not src_path.is_dir():
        print(f"⚠️ 오류: '{src_dir}' 디렉토리를 찾을 수 없거나 디렉토리가 아닙니다.")
        return

    # 2) 디렉토리 내부의 모든 .txt 파일 목록을 가져옴 (하위 디렉토리 미탐색)
    txt_files = list(src_path.glob("*.txt"))
    if len(txt_files) == 0:
        print(f"⚠️ '{src_dir}' 디렉토리에 .txt 파일이 하나도 없습니다.")
        return

    # 3) 모든 파일 내용을 합쳐서 하나의 큰 문자열로 만들기
    all_text = []
    print("📂 텍스트 파일 읽는 중...")
    for txt_file in tqdm(txt_files, desc="Reading .txt files", unit="file"):
        try:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {txt_file} → {e}")
            continue

        if len(lines) > 1:
            content = "\n".join(lines[1:]).strip()  # 두 번째 줄부터 끝까지 합침
            all_text.append(content)
        else:
            print(f"⚠️ '{txt_file.name}' 파일에 두 번째 줄이 없습니다. 건너뜁니다.")

    # 이제 all_text는 디렉토리 내 각 텍스트 파일의 내용을 담은 리스트
    # 이를 하나의 큰 문자열로 합칩니다.
    combined_text = "\n".join(all_text).strip()
    if not combined_text:
        print("⚠️ 주어진 디렉토리 내 모든 텍스트를 읽었으나, 내용이 비어있거나 모두 건너뛰었습니다.")
        return

    # 4) kss를 이용해 한글 문장 단위로 분리
    print("📝 kss 문장 분리 중...")
    try:
        sentences = kss.split_sentences(combined_text, backend='mecab')
    except Exception as e:
        print("❌ kss 문장 분리 중 오류 발생:", e)
        return

    # 5) CSV 파일 작성 (헤더: sentence,label)
    print("💾 CSV 생성 중...")
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

    print(f"✅ '{dst_csv}' 생성 완료 - 총 {count}개 문장 처리됨")

# -------------------------------------------------------------------
if __name__ == "__main__":
    """
    명령줄 인자를 통해 디렉토리 내 .txt 파일을 읽고 CSV 파일을 생성합니다.

    필수 인자:
        --src: 입력 디렉토리 경로
        --dst: 결과 파일 이름 접두어 (확장자는 자동으로 붙음)

    예시:
        python dataset.py --src raw_texts/ --dst train
    """
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
