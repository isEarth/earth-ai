"""
Trainer 상태에서 로그를 시각화하는 스크립트.

- HuggingFace Trainer의 checkpoint 디렉토리에서 가장 최신 `trainer_state.json`을 읽어,
  학습 및 검증 지표들을 시각화하고 그래프 이미지(.png)로 저장합니다.

- 사용 예시:
    python visualize_trainer_log.py

출력:
    - ./runs/.../plots/ 디렉토리에 그래프 저장
        - training_loss.png
        - eval_loss.png
        - eval_f1.png
        - ...
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 마지막 checkpoint 디렉토리 찾기
def find_latest_checkpoint(run_dir):
    """
    주어진 run 디렉토리에서 가장 최근의 checkpoint 디렉토리를 찾아 반환합니다.

    Args:
        run_dir (str): Trainer 실행 결과가 저장된 상위 디렉토리 경로

    Returns:
        str: 가장 최근 checkpoint 디렉토리 내부의 trainer_state.json 경로

    Raises:
        FileNotFoundError: checkpoint 디렉토리가 존재하지 않을 경우
    """
    checkpoints = glob.glob(os.path.join(run_dir, 'checkpoint-*'))
    if not checkpoints:
        raise FileNotFoundError("✅ checkpoint 디렉토리가 없습니다.")
    latest = max(checkpoints, key=os.path.getmtime)
    return os.path.join(latest, 'trainer_state.json')

# trainer_state.json 로드
def load_trainer_state(json_path):
    """
    trainer_state.json 파일을 불러와 log_history를 반환합니다.

    Args:
        json_path (str): trainer_state.json 파일 경로

    Returns:
        list[dict]: 로그 히스토리 리스트 (step 단위 기록 포함)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['log_history']

# 로그 데이터프레임화
def log_to_dataframe(log_history):
    """
    log_history 리스트를 pandas DataFrame으로 변환합니다.

    Args:
        log_history (list[dict]): 로그 딕셔너리 리스트

    Returns:
        pd.DataFrame: 로그를 정규화한 데이터프레임
    """
    df = pd.json_normalize(log_history)
    return df

# 그래프 그리기
def plot_metrics(df, output_dir):
    """
    로그 데이터프레임으로부터 학습 및 검증 지표를 그래프로 저장합니다.

    Args:
        df (pd.DataFrame): 로그 데이터프레임 (step, loss, eval_* 컬럼 포함)
        output_dir (str): 그래프 파일들을 저장할 디렉토리 경로

    Returns:
        None: 결과는 output_dir 내 PNG 파일들로 저장됨
    """
    os.makedirs(output_dir, exist_ok=True)

    # ① 학습 loss ---------------------------------------------------------
    if 'loss' in df.columns:
        plt.figure()
        (
            df.dropna(subset=['loss'])
              .plot(x='step', y='loss', title='Training Loss',)
        )
        plt.xlabel('step'); plt.ylabel('loss'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        # plt.close()

    # ② 검증 지표 ---------------------------------------------------------
    val_metrics = ['eval_loss', 'eval_f1', 'eval_precision',
                   'eval_recall', 'eval_roc_auc']

    for metric in val_metrics:
        if metric not in df.columns:
            continue

        plt.figure()
        ax = (
            df.dropna(subset=[metric])
              .plot(x='step', y=metric,
                    title=metric, marker='*', linestyle='-',
                    ax=plt.gca())      # 같은 figure에 그리기
        )

        # ──💬 각 점에 체크포인트 라벨 붙이기 ──────────────────────────────
        for _, row in df.dropna(subset=[metric]).iterrows():
            step = int(row['step'])            # 500, 1000, …
            yval = row[metric]
            label = f'ckpt-{step}'
            # 살짝 위(+5)로 올려서 겹침 방지
            ax.annotate(label,
                        xy=(step, yval),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8)

        plt.xlabel('step'); plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        # plt.close()

if __name__ == "__main__":
    """
    실행 블록: 가장 최근 checkpoint 디렉토리를 찾아 로그를 시각화합니다.
    - ./runs/run_*/checkpoint-*/trainer_state.json을 자동으로 감지
    - ./runs/run_*/plots/ 디렉토리에 그래프 파일 저장
    """
    run_dir = './runs/run_20250603_161825'  # 이 디렉토리 안에서 찾음
    json_path = find_latest_checkpoint(run_dir)
    print(f"✅ 읽어온 trainer_state.json 경로: {json_path}")

    log_history = load_trainer_state(json_path)
    df = log_to_dataframe(log_history)

    output_dir = os.path.join(run_dir, 'plots')
    plot_metrics(df, output_dir)

    print(f"✅ 그래프들이 {output_dir}에 저장되었습니다.")