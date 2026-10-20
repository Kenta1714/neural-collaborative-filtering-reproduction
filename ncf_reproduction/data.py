import os
import ast
from unittest import TestLoader
from xml.sax.handler import feature_namespace_prefixes
from matplotlib.rcsetup import validate_any
from networkx import neighbors
from numpy.random import RandomState

import torch


from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import scipy.sparse as sp

import logging

logger = logging.getLogger(__name__)


def build_validation_data(val_positives, item_num, train_mat, seed, num_negatives=99):
    """validationデータ作成用の関数

    Args:
        val_positives (_type_): _description_
        item_num (_type_): _description_
        train_mat (_type_): _description_
        seed (_type_): _description_
        num_negatives (int, optional): _description_. Defaults to 99.

    Returns:
        _type_: _description_
    """
    rng = np.random.default_rng(seed)
    val_data = []

    for user_id, pos_item_id in val_positives:
        val_data.append([user_id, pos_item_id])
        # すでに抽出した負例を重複除外用に記録しておく（test_dataでも負例同士の重複はなかったため除外する）
        sampled_items = set()

        for _ in range(num_negatives):
            item_id = int(rng.integers(item_num))
            # trainの正例と抽出済みの負例を除外して負例を抽出する
            while (user_id, item_id) in train_mat or item_id in sampled_items:
                item_id = int(rng.integers(item_num))

            sampled_items.add(item_id)
            val_data.append([user_id, item_id])
    return val_data


def load_data(
    base_dir,
    val_seed,
    dataset="ml-1m",
):
    # TODO: configに移してもいいかも
    train_rating = os.path.join(base_dir, "data", f"{dataset}.train.rating")
    # 今回はtest.negativeの最初の列を正例として扱うのでtest_ratingは使用しない。
    # test_rating = os.path.join(base_dir, "data", f"{dataset}.test.rating")
    test_negative = os.path.join(base_dir, "data", f"{dataset}.test.negative")

    # データはtsvファイルなのでsep="\t"で読み込む
    # また、headerなしなので自分でheaderを作成する
    # 公式実装リポジトリに以下の通り記載あり
    # rating:
    # userID\t itemID\t rating\t timestamp (if have)
    # negative:
    # Each line corresponds to the line of test.rating, containing 99 negative samples.
    # Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
    # train側ではuser列とitem列のみを取り出す
    original_train_df = pd.read_csv(
        train_rating,
        sep="\t",
        header=None,
        names=["user", "item"],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32},  # 各列でデータ型を指定している。
    )
    # REVIEW: 下記はIDが0から振られている前提？
    user_num = original_train_df["user"].max() + 1
    item_num = original_train_df["item"].max() + 1

    # 検証用データとして、各ユーザーからランダムに一件選ぶ（S4.1）
    val_df = original_train_df.groupby("user", sort=True).sample(
        n=1, random_state=val_seed
    )

    val_positives = val_df.values.tolist()
    # val_dataに抽出していない分をtrainデータとする
    train_data = original_train_df.drop(index=val_df.index).values.tolist()

    # user数 × item数の疎行列を作成
    # この行列はtrainでの負例サンプリング時に、正例を除くために使われている。
    # なお、公式実装でもtest_dataのinteractionはこの行列に含まれないため、負例としてtestの正例がサンプリングされる可能性がある設計となっていた。
    # よってvalidationでも同様にvalidationの正例自体は評価行列に含めずtrain_dataのみで除外用の行列を作成する
    train_mat = sp.dok_matrix((user_num, item_num))
    # train_dataの各行が0列目のuser idと1列目のitem idのinteractionを表すのでそこを1にする
    for user_id, item_id in train_data:
        train_mat[user_id, item_id] = 1.0

    # validation_dataを作成する
    val_data = build_validation_data(
        val_positives=val_positives,
        item_num=item_num,
        train_mat=train_mat,
        seed=val_seed,
    )

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
    logger.info("データの読み込みと加工完了。")
    logger.info(f"train_data: {len(train_data)}")
    logger.info(f"test_data: {len(test_data)}")
    logger.info(f"user_num: {user_num}")
    logger.info(f"train_mat: {train_mat.shape}")

    return train_data, val_data, test_data, user_num, item_num, train_mat


class NCFDataset(Dataset):
    def __init__(self, features, item_num, train_mat=None, num_ng=0, is_training=None):
        """_summary_

        Args:
            features (_type_): interactionした[user,item]のリスト
            item_num (_type_): _description_
            train_mat (_type_, optional): _description_. Defaults to None.
            num_ng (int, optional): 負例の個数？. Defaults to 0.
            is_training (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.features_ps = features
        self.item_num = item_num
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
                item_id = np.random.randint(self.item_num)
                # dok_matrixでは内部で{(m,n):1.0}のような辞書で値を保持しているためこのようなキーの存在判定が可能。train_mat[user,item]!=0と同じ挙動
                while (user_id, item_id) in self.train_mat:
                    item_id = np.random.randint(self.item_num)
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


def build_dataloaders(base_dir, batch_size, num_neg, num_workers, generator, val_seed):
    train_data, val_data, test_data, user_num, item_num, train_mat = load_data(
        base_dir=base_dir, val_seed=val_seed, dataset="ml-1m"
    )
    train_dataset = NCFDataset(
        features=train_data,
        item_num=item_num,
        train_mat=train_mat,
        num_ng=num_neg,
        is_training=True,
    )
    val_dataset = NCFDataset(
        features=val_data,
        item_num=item_num,
        train_mat=train_mat,
        # 元々validationには負例が含まれているから追加でサンプリングする負例数num_ngは0
        num_ng=0,
        is_training=False,
    )
    test_dataset = NCFDataset(
        features=test_data,
        item_num=item_num,
        train_mat=train_mat,
        # 元々testには負例が含まれているから追加でサンプリングする負例数num_ngは0
        num_ng=0,
        is_training=False,
    )

    # ローダー作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        generator=generator,
    )
    # test_dataについてはバッチを100個ずつ取る前提であることに注意
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        generator=generator,
    )
    return (train_loader, val_loader, test_loader, user_num, item_num)
