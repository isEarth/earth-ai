import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RVGAE(nn.Module):
    """
    RVGAE: Relational Variational Graph Autoencoder

    관계형 그래프 구조를 처리할 수 있는 VAE 기반 GNN 모델.
    - 인코더는 R-GCN을 사용하여 노드 임베딩의 평균과 분산을 학습
    - 디코더는 노드 쌍 임베딩을 기반으로 링크 존재 여부와 관계 유형을 예측

    Args:
        in_channels (int): 입력 피처 차원
        hidden_channels (int): 인코딩 중간 은닉 차원
        out_channels (int): 잠재 공간(z)의 차원
        num_relations (int): 엣지 관계 유형 개수
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RVGAE, self).__init__()
        self.num_relations = num_relations

        # === 인코더 (R-GCN 기반) ===
        self.rgcn_base = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn_mean = RGCNConv(hidden_channels, out_channels, num_relations)
        self.rgcn_logstd = RGCNConv(hidden_channels, out_channels, num_relations)

        # === 공유 디코더 ===
        self.shared_decoder = nn.Sequential(
            nn.Linear(2 * out_channels, 128),
            nn.ReLU()
        )

        # === 출력층 분기 ===
        self.link_out = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.type_out = nn.Linear(128, num_relations)  # 다중 클래스 분류

    def encode(self, x, edge_index, edge_type):
        """
        노드 임베딩을 정규분포의 평균 및 분산으로 인코딩

        Args:
            x (Tensor): 노드 피처 (num_nodes, in_channels)
            edge_index (Tensor): 엣지 인덱스 (2, num_edges)
            edge_type (Tensor): 관계 유형 인덱스 (num_edges,)

        Returns:
            z (Tensor): 샘플링된 노드 임베딩 (reparameterization)
            mean (Tensor): 정규분포 평균
            logstd (Tensor): 로그 표준편차
        """
        h = F.relu(self.rgcn_base(x, edge_index, edge_type))
        mean = self.rgcn_mean(h, edge_index, edge_type)
        logstd = self.rgcn_logstd(h, edge_index, edge_type)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logstd

    def decode(self, z, edge_index):
        """
        노드 임베딩 z를 사용하여 노드쌍 간 링크 예측 및 관계 유형 분류 수행

        Args:
            z (Tensor): 노드 임베딩 (num_nodes, out_channels)
            edge_index (Tensor): 예측할 노드쌍 인덱스 (2, num_edges)

        Returns:
            link_pred (Tensor): 링크 존재 확률 (num_edges,)
            type_pred (Tensor): 관계 유형 로짓 (num_edges, num_relations)
        """
        src = z[edge_index[0]] # => edge_index[0]는 (num_edges,)짜리 텐서
        dst = z[edge_index[1]] # => [edge_index[0]]는 (num_edges, embedding_dim)
        z_pair = torch.cat([src, dst], dim=1)  # shape: (batch_size, 2*out_dim)

        share = self.shared_decoder(z_pair)
        link_pred = self.link_out(share).squeeze()  # shape: (batch_size,)
        type_pred = self.type_out(share)          # shape: (batch_size, num_relations)

        return link_pred, type_pred

    def forward(self, x, edge_index, edge_type, pos_edge_index):
        """
        전체 RVGAE 순전파

        Args:
            x (Tensor): 노드 피처
            edge_index (Tensor): 전체 그래프 엣지
            edge_type (Tensor): 관계 유형 인덱스
            pos_edge_index (Tensor): 예측 대상 엣지 쌍 (양성 샘플)

        Returns:
            link_pred (Tensor): 링크 존재 예측
            type_pred (Tensor): 관계 유형 예측
            mean (Tensor): 인코더 평균
            logstd (Tensor): 인코더 로그 표준편차
            z (Tensor): 샘플링된 임베딩
        """
        z, mean, logstd = self.encode(x, edge_index, edge_type)
        link_pred, type_pred = self.decode(z, pos_edge_index)
        return link_pred, type_pred, mean, logstd, z