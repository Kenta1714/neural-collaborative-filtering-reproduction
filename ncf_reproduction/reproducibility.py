import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

import numpy as np

import random

# 再現性担保用の設定を入れる


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)

    # 現在は下記でCPUと全てのCUDAデバイスをseedする。よってtorch.cuda.manual_seed(seed)やtorch.cuda.manual_seed_all(seed)は不要。
    torch.manual_seed(seed)

    # 下記でpytorch全体で可能な限り決定的なアルゴリズムを使用させる設定。torch.backends.cudnn.deterministic = TrueはcuDNNに対してのみ適用されるのでこれより狭い。
    torch.use_deterministic_algorithms(True)

    # 複数の畳み込みアルゴリズムを実際に試して最速だったものを選ぶ機能を無効化。（つまりアルゴリズムを統一させる）
    torch.backends.cudnn.benchmark = False

    # dataloaderに対して、pytorch全体の乱数処理とは独立にシャッフル順序の乱数処理を行うことができる。ただしdataloaderの生成時にgenerator引数に渡す必要あり。
    g = torch.Generator().manual_seed(seed)

    return g

    # dataloaderにおけるnum_workersを1より大きくする場合は追加で設定が必要らしい
