"""
RVGAE (Relational Variational Graph Autoencoder)

이 모델은 노드 간 관계 정보를 학습하여:
- 노드 임베딩(z)을 생성하고
- 노드 쌍 간의 링크 존재 여부와
- 관계 타입(링크 유형)을 예측하는 목적의 그래프 오토인코더이다.

[구성요소]
1. 인코더 (R-GCN 기반): 
   - 입력 노드 임베딩(x)과 관계 유형(edge_type)을 이용해
   - 평균(mean), 표준편차(logstd)를 생성하고,
   - 샘플링된 잠재 벡터 z를 생성함

2. 디코더 (공유 구조):
   - 두 노드 간 임베딩을 concat → shared layer → 
   - [1] 링크 존재 확률 예측 (Sigmoid)
   - [2] 링크 타입 분류 (Softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Tuple

class RVGAE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_relations: int):
        """
        RVGAE 초기화 메서드

        Args:
            in_channels (int): 입력 노드 피처 차원 (F)
            hidden_channels (int): R-GCN 은닉층 차원
            out_channels (int): 최종 잠재 임베딩 z의 차원 (D)
            num_relations (int): 관계 타입 수 (다중 분류 클래스 수)
        """
        super(RVGAE, self).__init__()
        self.num_relations = num_relations

        # === 인코더 ===
        self.rgcn_base = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn_mean = RGCNConv(hidden_channels, out_channels, num_relations)
        self.rgcn_logstd = RGCNConv(hidden_channels, out_channels, num_relations)

        # === 공유 디코더 구조 ===
        self.shared_decoder = nn.Sequential(
            nn.Linear(2 * out_channels, 128),
            nn.ReLU()
        )

        # === 분기 출력층 ===
        # 링크 예측은 시그모이드를 거쳐 확률로 출력됨 (B,)
        self.link_out = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # 관계 타입 예측은 타입 중 하나로 출력됨(B, R)
        self.type_out = nn.Linear(128, num_relations)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        인코더: R-GCN을 통해 z 임베딩, 평균(mean), 로그표준편차(logstd) 생성

        Args:
            x (Tensor): 입력 노드 피처 (N, F)
            edge_index (Tensor): 엣지 인덱스 (2, E)
            edge_type (Tensor): 엣지 타입 (E,)

        Returns:
            z (Tensor): 샘플링된 노드 임베딩 (N, D)
            mean (Tensor): 평균 벡터 (N, D)
            logstd (Tensor): 로그 표준편차 벡터 (N, D)
        """
        h = F.relu(self.rgcn_base(x, edge_index, edge_type))
        mean = self.rgcn_mean(h, edge_index, edge_type)
        logstd = self.rgcn_logstd(h, edge_index, edge_type)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mean + eps * std  # reparameterization
        return z, mean, logstd

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        디코더: 노드 임베딩 z를 사용하여 링크 여부 및 관계 타입 예측

        Args:
            z (Tensor): 노드 임베딩 (N, D)
            edge_index (Tensor): 예측 대상 edge 쌍 (2, B)

        Returns:
            link_pred (Tensor): 링크 존재 확률 (B,)
            type_pred (Tensor): 관계 타입 출력 (B, R)
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        z_pair = torch.cat([src, dst], dim=1)

        share = self.shared_decoder(z_pair)
        link_pred = self.link_out(share).squeeze()  # (B,)
        type_pred = self.type_out(share)  # (B, R)
        return link_pred, type_pred

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, pos_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        전체 forward: z 임베딩 생성 → 양성 엣지에 대한 링크/관계 예측

        Args:
            x (Tensor): 노드 피처 (N, F)
            edge_index (Tensor): 전체 그래프 엣지 (2, E)
            edge_type (Tensor): 엣지 관계 타입 (E,)
            pos_edge_index (Tensor): 양성 엣지 인덱스 (2, E_pos)

        Returns:
            link_pred (Tensor): 양성 엣지에 대한 링크 예측 확률 (E_pos,)
            type_pred (Tensor): 양성 엣지에 대한 관계 타입 예측 출력 (E_pos, R)
            mean (Tensor): 평균 벡터 (N, D)
            logstd (Tensor): 로그 표준편차 (N, D)
            z (Tensor): 샘플링된 임베딩 (N, D)
        """
        z, mean, logstd = self.encode(x, edge_index, edge_type)
        link_pred, type_pred = self.decode(z, pos_edge_index)
        return link_pred, type_pred, mean, logstd, z
