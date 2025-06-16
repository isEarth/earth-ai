import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RVGAE(nn.Module):
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
        h = F.relu(self.rgcn_base(x, edge_index, edge_type))
        mean = self.rgcn_mean(h, edge_index, edge_type)
        logstd = self.rgcn_logstd(h, edge_index, edge_type)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logstd

    def decode(self, z, edge_index):
        """공통 디코더 → 링크 확률 + 링크 타입 예측"""
        src = z[edge_index[0]] # => edge_index[0]는 (num_edges,)짜리 텐서
        dst = z[edge_index[1]] # => [edge_index[0]]는 (num_edges, embedding_dim)
        z_pair = torch.cat([src, dst], dim=1)  # shape: (batch_size, 2*out_dim)

        share = self.shared_decoder(z_pair)
        link_pred = self.link_out(share).squeeze()  # shape: (batch_size,)
        type_pred = self.type_out(share)          # shape: (batch_size, num_relations)

        return link_pred, type_pred

    def forward(self, x, edge_index, edge_type, pos_edge_index): 
        z, mean, logstd = self.encode(x, edge_index, edge_type)
        link_pred, type_pred = self.decode(z, pos_edge_index)
        return link_pred, type_pred, mean, logstd, z