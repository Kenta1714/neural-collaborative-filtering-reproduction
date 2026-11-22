from email.mime import base
import os
import time

os.environ["TZ"] = "Asia/Tokyo"
time.tzset()

from logging.config import dictConfig
import logging

from pathlib import Path


import hydra
import wandb

import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ncf_reproduction.data import build_dataloaders

# from ncf_reproduction.engine import evaluate, fit_step_validation
from ncf_reproduction.trainer import Trainer
from ncf_reproduction.model import GMF, MLP, NeuMF

from ncf_reproduction.reproducibility import set_reproducibility

# from ncf_reproduction.visualization import plot_error_history

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # --------------------------------
    # 読み込まれた設定を表示
    # --------------------------------
    logger.info(f"{"="*10}Configuration{"="*10}")
    # 下記のto_yamlはlog用にYAML文字列に変換するもの。dictに変換するto_containerとは違うので注意。
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    # --------------------------------
    # 再現性設定
    # --------------------------------
    g = set_reproducibility(cfg.seed)

    # --------------------------------
    # デバイス決定
    # --------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --------------------------------
    # パスの準備
    # --------------------------------
    # yamlに書かれたパスが相対パスでも絶対パスに変換
    # 絶対パスならそのまま使う
    base_dir = Path(to_absolute_path(cfg.data.base_dir))

    # Hydraが現在のrunの実行結果を格納するために作成したディレクトリ（日付や時間入りのもの）
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # 上記で指定したディレクトリの中にcheck pointを保存する
    checkpoint_path = output_dir / cfg.checkpoint.filename

    # 上記で特定したディレクトリの中にlog.jpgを保存する
    # log_img_path = output_dir / "全体のlog.jpg"

    logger.info(f"base dir: {base_dir}")
    logger.info(f"output dir: {output_dir}")
    # --------------------------------
    # wandbに渡すためにcfgを辞書に変換する
    # --------------------------------
    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        # 未設定の値が残っていた場合にエラーにする
        throw_on_missing=True,
    )
    # wandb自身の設定はconfigから除外する
    wandb_config.pop("wandb", None)

    # パスの情報も除外する
    data_config = wandb_config.get("data")

    if isinstance(data_config, dict):
        data_config.pop("base_dir", None)

    # --------------------------------
    # wandb.init()を追加する（基本的にinitの引数にはyamlの値を渡す）
    # --------------------------------
    wandb_mode = cfg.wandb.mode if cfg.wandb.enabled else "disabled"

    with wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        mode=wandb_mode,
        config=wandb_config,
        dir=str(output_dir),
        job_type="train",
    ) as run:  # ここでrunオブジェクトができる
        # --------------------------------
        # wandbのrun設定
        # --------------------------------

        # 横軸をepochにする。デフォルトだと何回目のrun.log()呼び出しかが横軸になるので今回は明示しなくても同じプロットになる。
        run.define_metric("trainer/epoch")
        run.define_metric("train/*", step_metric="trainer/epoch")
        run.define_metric("val/*", step_metric="trainer/epoch")
        run.define_metric("optimizer/*", step_metric="trainer/epoch")

        # log送信用の関数をここで定義する。これをfitに渡す。
        def log_evaluation(metrics_dict):
            run.log(metrics_dict)

        # --------------------------------
        # dataloaderの作成
        # --------------------------------

        train_loader, val_loader, test_loader, user_num, item_num = build_dataloaders(
            base_dir=base_dir,
            batch_size=cfg.data.batch_size,
            num_neg=cfg.training.num_ng,
            num_workers=cfg.data.num_workers,
            generator=g,
            val_seed=cfg.seed,
        )

        # --------------------------------
        # モデルの作成
        # --------------------------------
        if cfg.model.model_name == "GMF":
            model = GMF(user_num, item_num, cfg.model.factor_num).to(device)

        if cfg.model.model_name == "MLP":
            model = MLP(
                user_num,
                item_num,
                cfg.model.factor_num,
                cfg.model.num_layers,
                cfg.model.dropout,
            ).to(device)

        if cfg.model.model_name == "NeuMF":
            model = NeuMF(
                user_num,
                item_num,
                cfg.model.factor_num,
                cfg.model.num_layers,
                cfg.model.dropout,
            ).to(device)

            if cfg.model.use_pretrain == True:
                # GMFとMLPをインスタンス化して保存した重みで上書きし、load_pretrainedに渡す
                gmf_model = GMF(user_num, item_num, cfg.model.factor_num)
                gmf_checkpoint = torch.load(
                    cfg.model.gmf_checkpoint_path,
                    map_location=device,
                    weights_only=True,
                )
                gmf_model.load_state_dict(gmf_checkpoint["model_state_dict"])
                mlp_model = MLP(
                    user_num,
                    item_num,
                    cfg.model.factor_num,
                    cfg.model.num_layers,
                    cfg.model.dropout,
                )
                mlp_checkpoint = torch.load(
                    cfg.model.mlp_checkpoint_path, map_location="cpu", weights_only=True
                )
                mlp_model.load_state_dict(mlp_checkpoint["model_state_dict"])
                model.load_pretrained(gmf_model, mlp_model)

        # --------------------------------
        # 損失関数の定義
        # --------------------------------
        criterion = nn.BCEWithLogitsLoss()

        # --------------------------------
        # optimizerの定義
        # --------------------------------
        if cfg.model.model_name == "NeuMF" and cfg.model.use_pretrain == True:
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

        # --------------------------------
        # 学習（Trainerクラスへの移行時にfit_step_validation関数の呼び出し処理から、Trainerクラスのインスタンス化処理＋fitメソッド呼び出しに変えた）
        # --------------------------------

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=cfg.training.epochs,
            top_k=cfg.evaluate.top_k,
            checkpoint_path=checkpoint_path,
            checkpoint_monitor=cfg.checkpoint.monitor,
            on_evaluation=log_evaluation,
        )

        history = trainer.fit()

        # --------------------------------
        # 最終評価（evaluate関数の呼び出しから、Trainerクラスのevaluateメソッド呼び出しに変えた。）
        # すでにtrainerインスタンスのmodel変数には学習済みのモデルが入っているのでこのようにできる
        # 上記に加えて、さらに最良チェックポイントを呼び出してそのモデルで評価するように変更した
        # --------------------------------

        # ptファイルからチェックポイントを読み込む
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # 読み込んだ重みをモデルのインスタンスに設定する（上書きということ？）
        trainer.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(
            "Best checkpoint loaded: "
            f"epoch={checkpoint['epoch']} "
            f"val_hr={checkpoint['best_val_hr']:.4f} "
            f"val_ndcg={checkpoint['best_val_ndcg']:.4f}"
        )

        # これで最良状態になったので、これで一回evaluate
        test_metrics = trainer.evaluate(
            dataloader=test_loader,
            max_batches=None,
        )

        logger.info(
            "Final test results: "
            f"HR@{cfg.evaluate.top_k}={test_metrics['hr']:.4f} "
            f"NDCG@{cfg.evaluate.top_k}={test_metrics['ndcg']:.4f}"
        )

        run.summary["test/hr"] = test_metrics["hr"]
        run.summary["test/ndcg"] = test_metrics["ndcg"]

        # # TODO 下記はコピペ
        # if history:
        #     best_val_error = min(record["val/error_rate"] for record in history)

        #     run.summary["best/val_error_rate"] = best_val_error


if __name__ == "__main__":
    main()
