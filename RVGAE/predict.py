"""
[RVGAE 예측 파이프라인 설명]

본 코드는 Heterogeneous grahp에서 노드 간 숨은 링크 및 타입을 예측하기 위해 학습 및 예측 파이프라인이다.

주요 목표:
- 고차원 임베딩 벡터를 기반으로 노드 간 링크 존재 여부 및 관계 유형(링크 타입)을 예측
- DeBERTa 기반 의미 임베딩과 S-bert의 cls를 각각 압축하여 생성된 임베딩과 Graph AutoEncoder를 결합하여 구조화된 의미 네트워크 추출

[구성 요소]
- RVGAE 모델 : 인코더(R-GCN 기반)와 디코더(MLP type)로 구성된 숨은 링크 및 타입 추론용 그래프 오토인코더
- load_data : 구절 임베딩, 노드 쌍, 노드 쌍 관계 유형 데이터를 로드하여 텐서 변환
- compute_class_weights : 관계 유형 불균형 문제를 해결하기 위한 가중치 계산
- generate_negative_edges : 그래프에 존재하지 않는 노드쌍 중 무작위로 음성(edge) 샘플링
- train : RVGAE 모델 학습 루프, reconstruction + KL + 타입 예측 손실 포함
- predict_links : 학습된 z 벡터를 이용해 GT에 없는 노드쌍 중에서 숨은 링크 및 관계 예측
- main : 전체 실행 흐름 제어 및 결과 저장
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from model import RVGAE
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    hidden_channels: int = 64
    out_channels: int = 32
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 100000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class FileName:
    node_feature: str = "./data/x.npy"
    edge_index: str = "./data/edge_index.npy"
    edge_type: str = "./data/edge_type.npy"

def load_data(filenames: FileName, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    노드 임베딩(x), 엣지 인덱스(edge_index), 엣지 타입(edge_type)을 파일에서 로드하고 텐서로 변환

    Args:
        filenames (FileName): 데이터 파일 경로 구조체
        device (str): 장치 이름 ('cuda' 또는 'cpu')

    Returns:
        Tuple[Tensor, Tensor, Tensor]: x, edge_index, edge_type (GPU에 로드됨)
    """
    x = torch.tensor(np.load(filenames.node_feature), dtype=torch.float)
    edge_index = torch.tensor(np.load(filenames.edge_index), dtype=torch.long)
    edge_type = torch.tensor(np.load(filenames.edge_type), dtype=torch.long)
    return x.to(device), edge_index.to(device), edge_type.to(device)

def compute_class_weights(edge_type: torch.Tensor, num_relations: int, device: str) -> torch.Tensor:
    """
    각 관계 타입의 클래스 불균형을 보정하기 위한 가중치 계산

    Args:
        edge_type (Tensor): 각 엣지의 관계 타입 (E,)
        num_relations (int): 관계 타입 개수
        device (str): 장치 이름

    Returns:
        Tensor: 관계 타입별 정규화된 가중치 벡터 (R,)
    """
    with torch.no_grad():
        counts = torch.bincount(edge_type, minlength=num_relations).float()
        weights = 1.0 / (torch.log1p(counts) + 1e-6)
        weights = weights / weights.sum()
        return weights.to(device)

def generate_negative_edges(x: torch.Tensor, pos_edge_index: torch.Tensor) -> torch.Tensor:
    """
    그래프에 존재하지 않는 노드쌍 중 무작위로 음성(edge) 샘플링

    Args:
        x (Tensor): 노드 피처 (N, F)
        pos_edge_index (Tensor): 존재하는 양성 엣지 인덱스 (2, E)

    Returns:
        Tensor: negative edge 인덱스 (2, E)
    """
    num_nodes = x.size(0)
    num_pos_edges = pos_edge_index.size(1)
    all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    pos_edges_set = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
    negative_candidates = [pair for pair in all_pairs if pair not in pos_edges_set]
    random.seed(42)
    neg_samples = random.sample(negative_candidates, num_pos_edges)
    neg_edge_index = torch.tensor(neg_samples, dtype=torch.long).t()
    return neg_edge_index

def train(model: RVGAE, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, class_weights: torch.Tensor, config: Config):
    """
    RVGAE 모델 학습 루프

    Args:
        model (RVGAE): 학습할 모델
        x (Tensor): 노드 피처 (N, F)
        edge_index (Tensor): 엣지 인덱스 (2, E)
        edge_type (Tensor): 엣지 타입 (E,)
        class_weights (Tensor): 클래스 가중치 (R,)
        config (Config): 학습 설정값
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    pos_edge_index = edge_index
    neg_edge_index = generate_negative_edges(x, pos_edge_index).to(config.device)

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        _, _, mean, logstd, z = model(x, edge_index, edge_type, pos_edge_index)
        pos_out, _ = model.decode(z, pos_edge_index)
        neg_out, _ = model.decode(z, neg_edge_index)

        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        kl_loss = -0.5 / x.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mean**2 - torch.exp(2 * logstd), dim=1))
        _, type_pred = model.decode(z, edge_index)
        type_loss = F.cross_entropy(type_pred, edge_type, weight=class_weights)

        loss = pos_loss + neg_loss + kl_loss + type_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {loss.item():.4f}")

def predict_links(model: RVGAE, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, config: Config, threshold: float = 0.6) -> List[Tuple[int, int, float, int]]:
    """
    학습된 RVGAE 모델을 사용하여 숨은 링크 및 관계 타입을 예측

    Args:
        model (RVGAE): 학습 완료된 모델
        x (Tensor): 노드 임베딩 입력 (N, F)
        edge_index (Tensor): GT에 포함된 실제 엣지 인덱스 (2, E)
        edge_type (Tensor): 엣지 관계 타입 (E,)
        config (Config): 배치 사이즈 포함 학습 설정
        threshold (float): 연결로 간주할 최소 확률값

    Returns:
        List[Tuple[int, int, float, int]]: [(source, target, score, predicted_relation)]
    """
    model.eval()
    with torch.no_grad():
        _, _, _, _, z = model(x, edge_index, edge_type, edge_index)
        num_nodes = x.size(0)
        all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        gt_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        predict_pairs = [pair for pair in all_pairs if pair not in gt_pairs]

        results = []
        for i in range(0, len(predict_pairs), config.batch_size):
            batch = predict_pairs[i:i + config.batch_size]
            edge_batch = torch.tensor(batch, dtype=torch.long).t().to(x.device)
            link_pred, type_pred = model.decode(z, edge_batch)
            scores = link_pred.cpu().numpy()
            pred_types = torch.argmax(type_pred, dim=1).cpu().numpy()

            for (n1, n2), s, rel in zip(batch, scores, pred_types):
                if s >= threshold:
                    results.append((int(n1), int(n2), float(s), int(rel)))

    return results

def main():
    """
    전체 파이프라인 실행: 데이터 로딩 → 모델 학습 → 링크 예측 및 저장
    """
    config = Config()
    filenames = FileName()
    x, edge_index, edge_type = load_data(filenames, config.device)
    num_relations = int(edge_type.max().item()) + 1
    model = RVGAE(x.size(1), config.hidden_channels, config.out_channels, num_relations).to(config.device)

    class_weights = compute_class_weights(edge_type, num_relations, config.device)
    train(model, x, edge_index, edge_type, class_weights, config)
    results = predict_links(model, x, edge_index, edge_type, config)
    df = pd.DataFrame(results, columns=["source", "target", "score", "predicted_relation"])
    df.to_csv("predicted_links.csv", index=False)
    print("예측 결과가 'predicted_links.csv'로 저장되었습니다.")

if __name__ == "__main__":
    main()
