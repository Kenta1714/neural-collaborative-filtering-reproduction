import os
import ast
from xml.sax.handler import feature_namespace_prefixes
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
    for user_id, item_id in train_data:
        train_mat[user_id, item_id] = 1.0

    # test_dataについては概ね参考リポジトリ通り
    test_data = []
    with open(test_negative, "r") as fd:
        line = fd.readline()
        while line != None and line != "":
            arr = line.split("\t")
            # 0列目のみuser,itemのタプルが文字列として入っているのでuserとitemを取り出す
            # 参考リポジトリはeval()を使用していたがコードを実行する必要がないのでast.literal_evalとした。
            user_id, pos_item_id = ast.literal_eval(arr[0])
            test_data.append([user_id, pos_item_id])
            # HACK: 1列目以降は負例itemなので、user,負例itemを99行分作成。後で100件を塊で取り出す想定なのでフラグ列が用意されていない
            for neg_item_id in arr[1:]:
                test_data.append([user_id, int(neg_item_id)])
            line = fd.readline()

    # 形状の確認
    logger.info("データの読み込みと加工完了。shapeは下記の通り。")
    logger.info(f"train_data: {train_data.shape}")
    logger.info(f"test_data: {test_data.shape}")
    logger.info(f"user_num: {user_num.shape}")
    logger.info(f"train_mat: {train_mat.shape}")

    return train_data, test_data, user_num, item_num, train_mat


class NCFDataset(Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        """_summary_

        Args:
            features (_type_): interactionした[user,item]のリスト？
            num_item (_type_): _description_
            train_mat (_type_, optional): _description_. Defaults to None.
            num_ng (int, optional): 負例の個数？. Defaults to 0.
            is_training (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        # HACK: test時はlabelは使わないが、user,item,labelというインターフェースを共通にするためにtest時はこれがダミーlabelとして返される
        self.labels = [0 for _ in range(len(features))]

    def sample_negatives(self):
        # HACK: 参考実装がややわかりにくいので要修正。また公式実装も含めて重複が許容されていることに注意。また、test positiveを負例としてサンプリングする可能性もある。（ただしこれを除外するのは）
        """testデータには負例が最初からセットになっているがtrainではここで作成する。エポックごとに呼ぶことで異なる負例を追加する想定"""
        assert self.is_training, "test時は負例のサンプリングは不要です"

        self.features_ng = []
        for user_id, _ in self.features_ps:
            # 設定している負例の個数分ループ
            for _ in range(self.num_ng):
                # itemのidをランダムに選択して疎行列に含まれていなかったらinteractionしていないので負例として採用する
                item_id = np.random.randint(self.num_item)
                # dok_matrixでは内部で{(m,n):1.0}のような辞書で値を保持しているためこのようなキーの存在判定が可能。train_mat[user,item]!=0と同じ挙動
                while (user_id, item_id) in self.train_mat:
                    item_id = np.random.randint(self.num_item)
                self.features_ng.append([user_id, item_id])
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_full = self.features_ps + self.features_ng
        self.labels_full = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_full if self.is_training else self.features_ps
        labels = self.labels_full if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label
