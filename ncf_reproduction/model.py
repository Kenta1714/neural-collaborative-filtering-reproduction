from turtle import forward

from sympy import factor
import torch
import torch.nn as nn
from yaml import emit


class GMFEncoder(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super().__init__()

        # nn.Embeddingで第一引数のそれぞれに第二引数の次元の埋め込みベクトルを割り当てて保持する。
        # ここでは各user idに対してfactor_num次元のベクトルが割り当てられた埋め込み行列となる
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)

        self._init_weights()

    def _init_weights(self):
        # 平均0標準偏差0.01のガウス分布で初期化(S4.1)。公式実装でも確認
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        return embed_user_GMF * embed_item_GMF


class MLPEncoder(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        super().__init__()
        self.dropout = dropout

        # nn.Embeddingで第一引数のそれぞれに第二引数の次元の埋め込みベクトルを割り当てて保持する。
        # layerの数だけ次元数が1/2倍されてfactor_num次元のベクトルを作りたいので元の特徴量はfactor_num2**num_layersにする必要がある
        # 実際のMLPの入力はuserとitemのconcatなので、Embeddingの次元数としては2をかける回数はnum_layersより一回分少ない
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1))
        )

        layers = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Linear(input_size, input_size // 2))
            layers.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # 平均0標準偏差0.01のガウス分布で初期化(S4.1)。公式実装でも確認
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        # MLPの中のnn.Linearについて初期化を設定
        # weightについては公式実装でKerasのglorot_uniformが使われており、参考再現実装では対応するxavier_uniform_が使われている。
        # biasは両方の実装で全て0で初期化されている。
        for module in self.MLP_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        return self.MLP_layers(torch.cat((embed_user_MLP, embed_item_MLP), -1))


class GMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num):

        super().__init__()

        self.gmf_encoder = GMFEncoder(user_num, item_num, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)
        # 下記二つの初期化はGMF、MLP、NeuMFで共通
        # 公式実装ではKerasのlecun_uniformが使われており、参考再現実装では対応する下記が使われている。
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")
        nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user, item):
        # [batch_size, 1]の二次元配列になっているので、criterionで扱いやすいようにlabelと同じく一次元配列に直す。
        return self.predict_layer(self.gmf_encoder(user, item)).view(-1)


class MLP(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):

        super().__init__()
        self.dropout = dropout

        self.mlp_encoder = MLPEncoder(
            user_num, item_num, factor_num, num_layers, dropout
        )
        self.predict_layer = nn.Linear(factor_num, 1)

        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")
        # 参考再現実装ではself.modules()を使用してMLP層と予測層の全てのbiasをまとめて0で初期化しているが、今回の実装では予測層はここで初期化
        nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user, item):
        return self.predict_layer(self.mlp_encoder(user, item)).view(-1)


class NeuMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):

        super().__init__()
        self.dropout = dropout
        self.gmf_encoder = GMFEncoder(user_num, item_num, factor_num)
        self.mlp_encoder = MLPEncoder(
            user_num, item_num, factor_num, num_layers, dropout
        )
        # 二つの表現をconcatしたものを予測層に渡すため*2
        self.predict_layer = nn.Linear(factor_num * 2, 1)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")
        nn.init.zeros_(self.predict_layer.bias)

    def load_pretrained(self, gmf: GMF, mlp: MLP):
        # GMFとMLPのencoderをコピーする
        self.gmf_encoder.load_state_dict(gmf.encoder.state_dict())
        self.mlp_encoder.load_state_dict(mlp.encoder.state_dict())

    def forward(self, user, item):
        concated = torch.cat((GMFEncoder(user, item), MLPEncoder(user, item)), -1)
        return self.predict_layer(concated).view(-1)
