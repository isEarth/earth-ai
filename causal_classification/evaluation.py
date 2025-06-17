"""
plot_metrics_from_trainer.py

HuggingFace Trainer를 사용한 학습 결과 중 `trainer_state.json` 파일을 분석하여,
학습 손실 및 검증 지표(e.g., loss, f1, precision 등)의 추이를 시각화합니다.

입력 구조:
    ./runs/run_YYYYMMDD_HHMMSS/checkpoint-*/trainer_state.json

출력:
    학습 및 검증 지표 그래프 (PNG 파일들)가 run 디렉토리 아래 `plots/`에 저장됩니다.

사용 예시:
    $ python plot_metrics_from_trainer.py

함수 구성:
    - find_latest_checkpoint(run_dir) → 최신 checkpoint 디렉토리 탐색
    - load_trainer_state(json_path) → trainer_state.json에서 log history 로드
    - log_to_dataframe(log_history) → pandas DataFrame으로 변환
    - plot_metrics(df, output_dir) → step별 loss 및 validation metrics 시각화
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 마지막 checkpoint 디렉토리 찾기
def find_latest_checkpoint(run_dir):
    checkpoints = glob.glob(os.path.join(run_dir, 'checkpoint-*'))
    if not checkpoints:
        raise FileNotFoundError("✅ checkpoint 디렉토리가 없습니다.")
    latest = max(checkpoints, key=os.path.getmtime)
    return os.path.join(latest, 'trainer_state.json')

# trainer_state.json 로드
def load_trainer_state(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['log_history']

# 로그 데이터프레임화
def log_to_dataframe(log_history):
    df = pd.json_normalize(log_history)
    return df

# 그래프 그리기
def plot_metrics(df, output_dir):
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
    run_dir = './runs/run_20250603_161825'  # 이 디렉토리 안에서 찾음
    json_path = find_latest_checkpoint(run_dir)
    print(f"✅ 읽어온 trainer_state.json 경로: {json_path}")

    log_history = load_trainer_state(json_path)
    df = log_to_dataframe(log_history)

    output_dir = os.path.join(run_dir, 'plots')
    plot_metrics(df, output_dir)

    print(f"✅ 그래프들이 {output_dir}에 저장되었습니다.")