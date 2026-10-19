import numpy as np
import torch


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return 1.0 / np.log2(index + 2)
    return 0


def metrics(model, test_loader, top_k, device):
    HR, NDCG = [], []
    # 1バッチでは1ユーザーの候補のみが入り、itemは先頭が正解であることを前提としている
    for user, item, _ in test_loader:
        user = user.to(device)
        item = item.to(device)

        # 対象userに割り当てられた正例1件と負例99件に対して予測を行う
        predictions = model(user, item)
        # torch.topkは値のテンソルとindexのテンソルを返す
        _, indices = torch.topk(predictions, top_k)
        recommends = item[indices].cpu().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    # torch.loadでweights_only=Trueに対応するためにfloat型にする
    return float(np.mean(HR)), float(np.mean(NDCG))
