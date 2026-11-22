import math
import torch
from tqdm import tqdm
import logging

from ncf_reproduction.evaluate import metrics

logger = logging.getLogger(__name__)

# モデルにあまり依存しない学習と評価の進め方をおく
# 元々はplot_errorを一定stepのfor文の中でこのファイルで呼び出し、一定stepごとにグラフを上書きしていたが、このファイルはhistoryの返却のみにし、train.py側で最後に一回だけグラフ描画する形に変更した

# 基本的に関数版からTrainerクラスへの移行で変わるのは、モデルの設定を全てインスタンス変数で管理するのでそれらの変数にself.がつくことくらい。
# あとは一応fit_step_validation関数をfitメソッドにした上で、その中の内側のforループのメイン処理を_train_one_batchとして切り出した。


class Trainer:
    # initの引数はfit_step_validationと全く同じ
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs,
        top_k,
        checkpoint_path,
        checkpoint_monitor,
        # 下記はlogをwandbに送る関数を渡すことを想定。engine.pyから直接行わないのはengine.pyとwandbeを独立させるため
        on_evaluation=None,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.top_k = top_k
        self.checkpoint_path = checkpoint_path
        self.checkpoint_monitor = checkpoint_monitor
        self.on_evaluation = on_evaluation

        # 今まではfit_step_validation関数内の先頭で定義していたが、インスタンス変数として保持する
        self.history = []
        self.global_step = 0

        # 最良hr保存用。大きいほどいいので最初は0にしておく
        self.best_val_hr = 0

    def _train_one_epoch(self, train_loader):
        self.model.train()
        # TODO: loaderのdataset属性？
        train_loader.dataset.sample_negatives()

        loss_sum = 0.0
        sample_count = 0

        for user, item, label in train_loader:
            user = user.to(self.device, non_blocking=True)
            item = item.to(self.device, non_blocking=True)
            label = label.float().to(self.device, non_blocking=True)

            # 勾配の初期化
            self.optimizer.zero_grad(set_to_none=True)

            prediction = self.model(user, item)
            loss = self.criterion(prediction, label)

            loss.backward()
            self.optimizer.step()

            # runnningの指標計算用
            batch_size = label.numel()
            # lossは平均されているので一旦総和に戻す（平均のまま保持して最後に平均の平均にしないのは、最後のバッチだけサンプル数が少なくなることがあり、それを同じ重みで扱うと正確な平均ではなくなってしまうから。）
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
        return loss_sum / sample_count

    @torch.no_grad()
    def evaluate(self, dataloader, max_batches=None):
        """データローダー全体に対して評価を行い、モデルのモードを元の状態に戻す

        Args:
            model (_type_): 評価対象モデル
            dataloader (_type_): 評価対象データローダー
            criterion (_type_): 損失関数
            device (_type_): CPUまたはGPU
            max_batches (_type_, optional): 評価に使用する最大バッチ数。Noneの場合はデータローダー全体を評価. Defaults to None.

        Returns:
            _type_: 評価指標を格納した辞書
        """

        # evaluate関数を適用する前のモデルの状態を記録しておく。
        # evaluate関数でmodel.eval()をやりっぱなしでも呼び出し側で明示的にmodel.train()に戻せば問題はないが、evaluate側で責任をもって元の状態に戻すところまでをやるという設計思想。
        was_training = self.model.training

        # evalモードにする
        self.model.eval()

        # metrics関数側でtest_loaderをループして平均を取るのでこちらでは関数の呼び出しだけでok
        hr, ndcg = metrics(self.model, dataloader, self.top_k, self.device)

        # # 概念的にはevaluateしたいデータ全体を一気に評価でいいが、メモリなどの観点で仕方なくfor文にしているイメージ。
        # for batch_idx, (images, labels) in enumerate(dataloader):
        #     # 指定されたバッチ数だけ評価する場合
        #     if max_batches is not None and batch_idx >= max_batches:
        #         break
        #     # nonblockignって何ですか
        #     # deviceに送る
        #     images = images.to(self.device, non_blocking=True)
        #     labels = labels.to(self.device, non_blocking=True)

        #     outputs = self.model(images)
        #     loss = self.criterion(outputs, labels)

        #     batch_size = images.size(0)

        #     # lossは平均されているので一旦総和に戻す（平均のまま保持して最後に平均の平均にしないのは、最後のバッチだけサンプル数が少なくなることがあり、それを同じ重みで扱うと正確な平均ではなくなってしまうから。）
        #     total_loss += loss.item() * batch_size
        #     total_samples += batch_size

        #     predicted_label = outputs.argmax(dim=1)
        #     total_correct += (predicted_label == labels).sum().item()

        # 評価が終わったら元の状態に戻す。
        self.model.train(was_training)

        # trainにもvalにも使用するのでtrainとかはキー名につけない
        return {
            # "loss": avg_loss,
            "hr": hr,
            "ndcg": ndcg,
        }

    def _save_checkpoint_if_best(self, log):
        """_summary_

        Args:
            log (_type_): 指標等を記録した辞書（fit関数の中で作成）

        Returns:
            _type_: _description_
        """
        # monitor=
        current_hr = log[self.checkpoint_monitor]

        # 今までの最良より小さいhrなら保存しない
        if current_hr <= self.best_val_hr:
            return False

        # そうでない場合は更新する
        self.best_val_hr = current_hr
        best_ndcg = log["val/ndcg"]

        # checkpoint自体はただの辞書。state_dict()はモデルの重みなどを返す
        # 学習の中断・再開も可能にしたい場合はoptimizerなどのstate_dictも保存する
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.epoch,
            "best_val_hr": current_hr,
            "best_val_ndcg": best_ndcg,
        }

        # torch.save(self.model,path)のようにモデル本体を保存する方法もあるが、state_dict()を保存する方が扱いやすい
        torch.save(checkpoint, self.checkpoint_path)

        logger.info(
            "Best checkpoint saved: "
            f"epoch={self.epoch} "
            f"val_hr={current_hr:.4f} "
            f"val_ndcg={best_ndcg:.4f}"
        )
        return True

    def fit(self):

        # fitをもう一度呼んだ場合も最初から学習履歴を作る？
        self.history = []
        self.epoch = 0

        # # 今回はエポック数ではなくステップ数での指定なので、最大ステップ数 / 合計ミニバッチ取得数でエポック数を算出する
        # # ceilは切り上げ
        # max_epochs = math.ceil(self.max_steps / len(self.train_loader))

        # with句を抜けた時に必ずプログレスバーのオブジェクトが閉じられる。
        with tqdm(dynamic_ncols=True, desc="Training") as pbar:
            for epoch in range(self.epochs):
                self.model.train()

                train_loss = self._train_one_epoch(self.train_loader)
                val_metrics = self.evaluate(self.val_loader)

                self.epoch += 1
                pbar.update(1)

                log = {
                    "trainer/epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/hr": val_metrics["hr"],
                    "val/ndcg": val_metrics["ndcg"],
                }

                # validationごとに、最良だった場合はcheck pointを保存する
                self._save_checkpoint_if_best(log)

                self.history.append(log)

                # ここでwandbへのlogの送信を行う。on_evaluationは辞書を受け取り、wandbへそのlogを送信する関数が渡される想定。
                if self.on_evaluation is not None:
                    self.on_evaluation(log)

                # TODO コピペ

                pbar.set_postfix(
                    ValHR=f"{log['val/hr']:.4f}",
                    ValNDCG=f"{log['val/ndcg']:.4f}",
                    TrainLoss=f"{log['train/loss']:.4f}",
                )

                logger.info(
                    f"\n"
                    f"Epoch: {epoch + 1} | "
                    f"train_loss: "
                    f"{log['train/loss']:.4f} | "
                    f"val_hr: "
                    f"{log['val/hr']:.4f} | "
                    f"val_ndcg: {log['val/ndcg']:.4f} | "
                )
                # 上記までコピペ

                # evaluate()内で責任を持って元のモードに戻しているが、念の為学習ループであることを明確にするため明示
                self.model.train()

        return self.history
