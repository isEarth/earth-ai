import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import RVGAE

def predict():
    x = torch.tensor(np.load("/home/billgates/workspace/RVGAE/data/x.npy"), dtype=torch.float)
    pos_edge_index = torch.tensor(np.load("/home/billgates/workspace/RVGAE/data/edge_index.npy"), dtype=torch.long)
    # neg_edge_index = torch.tensor(np.load("/home/billgates/workspace/RVGAE/data/neg_edge_index.npy"), dtype=torch.long)
    edge_type = torch.tensor(np.load("/home/billgates/workspace/RVGAE/data/edge_type.npy"), dtype=torch.long)

    edge_index = pos_edge_index
    in_channels = x.size(1)
    hidden_channels = 64
    out_channels = 32
    num_relations = int(edge_type.max().item()) + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RVGAE(in_channels, hidden_channels, out_channels, num_relations).to(device)

    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    pos_edge_index = pos_edge_index.to(device)
    # neg_edge_index = neg_edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # === edge_type 기반 가중치 자동 계산 ===
    with torch.no_grad():
        edge_type_counts = torch.bincount(edge_type, minlength=num_relations).float()
        class_weights = 1.0 / (torch.log1p(edge_type_counts) + 1e-6)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)

    # === 학습 루프 ===
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        _,_, mean, logstd, z = model(x, edge_index, edge_type, pos_edge_index)
        pos_out,_ = model.decode(z, pos_edge_index)
        # neg_out,_ = model.decode(z, neg_edge_index)

        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        # neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))

        kl_loss = -0.5 / x.size(0) * torch.mean(torch.sum(
            1 + 2 * logstd - mean**2 - torch.exp(2 * logstd), dim=1))

        # === 링크 타입 분류 손실 ===
        _, type_pred = model.decode(z, edge_index)
        type_loss = F.cross_entropy(type_pred, edge_type, weight=class_weights)

        # loss = pos_loss + neg_loss + kl_loss + type_loss
        loss = pos_loss + kl_loss + type_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # === 숨은 링크 + 타입 예측 ===
    model.eval()
    with torch.no_grad():
        _, _, _, _, z = model(x, edge_index, edge_type, pos_edge_index)

        num_nodes = x.size(0)
        pairs_a = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j] 
        pairs_b = [(j, i) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        all_pairs = pairs_a + pairs_b
        gt_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        
        predict_pairs = [pair for pair in all_pairs if pair not in gt_pairs]
        
        batch_size = 100000
        results = []

        for i in range(0, len(predict_pairs), batch_size):
            batch = predict_pairs[i:i + batch_size]
            edge_batch = torch.tensor(batch, dtype=torch.long).t().to(device)  # shape: (2, B)

            link_pred, type_pred = model.decode(z, edge_batch)
            scores = link_pred.cpu().numpy()
            pred_types = torch.argmax(type_pred, dim=1).cpu().numpy()

            for (n1, n2), s, rel in zip(batch, scores, pred_types):
                if s >= 0.6:
                    results.append((int(n1), int(n2), float(s), int(rel)))  # only high-score predictions

    return results


    # # === 모델 저장 ===
    # torch.save(model.state_dict(), "trained_rvgae.pt")

    # === 숨은 링크 + 타입 예측 ===
    # model.eval()
    # with torch.no_grad():
    #     _, _, _, _, z = model(x, edge_index, edge_type, pos_edge_index)

    #     num_nodes = x.size(0)
    #     pairs_a = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j] 
    #     pairs_b = [(j, i) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    #     all_pairs = pairs_a + pairs_b
    #     gt_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        
    #     predict_pairs = [pair for pair in all_pairs if pair not in gt_pairs]
        
    #     batch_size = 100000
    #     results = []

    #     for i in range(0, len(predict_pairs), batch_size):
    #         batch = predict_pairs[i:i + batch_size]
    #         edge_batch = torch.tensor(batch, dtype=torch.long).t().to(device)  # shape: (2, B)

    #         scores,_ = model.decode(z, edge_batch)
    #         scores_sigmoid = scores.cpu().numpy()

    #         high_score_mask = scores >= 0.6
    #         if high_score_mask.sum() > 0:
    #             edge_high = edge_batch[:, high_score_mask]
    #             _, type_pred = model.decode(z, edge_high)
    #             pred_types = torch.argmax(type_pred, dim=1).cpu().numpy()
    #         else:
    #             pred_types = []

    #         # idx = 0
    #         # for (n1, n2), s in zip(batch, scores_sigmoid):
    #         #     if s >= 0.6:
    #         #         rel = pred_types[idx]
    #         #         idx += 1
    #         #     else:
    #         #         rel = -1
    #         #     results.append((n1, n2, s, rel))
    #         idx = 0
    #         for (n1, n2), s in zip(batch, scores.cpu().numpy()):
    #             if s >= 0.6:
    #                 rel = int(pred_types[idx])
    #                 idx += 1
    #                 results.append((int(n1), int(n2), float(s), rel)) # shape: (N,4)
    # return results
