import os
import ast
from networkx import neighbors
from pyparsing import line
import torch


import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import scipy.sparse as sp

import logging

logger = logging.getLogger(__name__)


def load_data(base_dir, dataset="ml-1m"):
    # TODO: configに移してもいいかも
    train_rating = os.path.join(base_dir, "Data", f"{dataset}.train.rating")
    # 今回はtest.negativeの最初の列を正例として扱うのでtest_ratingは使用しない。
    # test_rating = os.path.join(base_dir, "Data", f"{dataset}.test.rating")
    test_negative = os.path.join(base_dir, "Data", f"{dataset}.test.negative")

    # データはtsvファイルなのでsep="\t"で読み込む
    # また、headerなしなので自分でheaderを作成する
    # 公式実装リポジトリに以下の通り記載あり
    # rating:
    # userID\t itemID\t rating\t timestamp (if have)
    # negative:
    # Each line corresponds to the line of test.rating, containing 99 negative samples.
    # Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
    # train側ではuser列とitem列のみを取り出す
    train_data = pd.read_csv(
        train_rating,
        sep="\t",
        header=None,
        names=["user", "item"],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32},  # 各列でデータ型を指定している。
    )
    # REVIEW: 下記はIDが0から振られている前提？
    user_num = train_data["user"].max() + 1
    item_num = train_data["item"].max() + 1

    train_data = train_data.values.tolist()

    # user数 × item数の疎行列を作成
    train_mat = sp.dok_matrix((user_num, item_num))
    # train_dataの各行が0列目のuser idと1列目のitem idのinteractionを表すのでそこを1にする
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # test_dataについては概ね参考リポジトリ通り
    test_data = []
    with open(test_negative, "r") as fd:
        line = fd.readline()
        while line != None and line != "":
            arr = line.split("\t")
            # 0列目のみuser,itemのタプルが文字列として入っているのでuserとitemを取り出す
            # 参考リポジトリはeval()を使用していたがコードを実行する必要がないのでast.literal_evalとした。
            user, item = ast.literal_eval(arr[0])
            test_data.append([user, item])
            # HACK: 1列目以降は負例itemなので、user,負例itemを99行分作成。後で100件を塊で取り出す想定なのでフラグ列が用意されていない
            for neg_item_id in arr[1:]:
                test_data.append([user, int(neg_item_id)])
            line = fd.readline()

    # 形状の確認
    logger.info("データの読み込みと加工完了。shapeは下記の通り。")
    logger.info(f"train_data: {train_data.shape}")
    logger.info(f"test_data: {test_data.shape}")
    logger.info(f"user_num: {user_num.shape}")
    logger.info(f"train_mat: {train_mat.shape}")

    return train_data, test_data, user_num, item_num, train_mat
