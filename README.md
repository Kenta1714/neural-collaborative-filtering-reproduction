# Neural Collaborative Filtering の再現と人気度の偏りの改善

MovieLens 1M で NeuMF を再現し、人気作品に偏った推薦を **Smooth xQuAD** で調整する。
[実行結果付き notebook](notebooks/neumf_pre_full_catalog_story.ipynb) に、推論から図の作成までをまとめた。

## 再現実装にあたっての注意点

参考にしたリポジトリは次の2つ。

- [公式実装：hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)：モデルの構成、初期化、事前学習の方法を確認。
- [PyTorch 版再現実装：guoyang9/NCF](https://github.com/guoyang9/NCF)：PyTorch での実装と学習設定を参考にした。

主に PyTorch 版から、次の点を変更した。

- **モデルのクラスを整理**：1つの `NCF` クラス内での切り替えを、`GMF`・`MLP`・`NeuMF` と共通の Encoder に分け、構成と重みの引き継ぎを読み取りやすくした。
- **val と test を分離**：学習データから各ユーザー1件を val に取り分け、val の HR でモデルを選択する。test は選択したモデルの最終評価に使う。

本編のモデルは参考実装の設定をもとに固定し、**モデルのハイパーパラメータ探索は行っていない**。
埋め込み次元の基準値32、MLP 3層、学習率0.001、バッチサイズ256、学習時の負例数4を使用した。
後述する再ランキングの係数 λ だけは val で選ぶ。

再現確認は「正例1件＋負例99件」、以下の追加実験は**全3,706作品から学習済み作品を除いた候補**で評価する。

## 1. 人気度の偏り

人気度を学習データ内の出現回数で定義し、累積出現回数の約80%を占める作品を「人気作品」、残りを「低人気作品」とする。
ユーザーの履歴では低人気作品が平均 **15.88%** を占める一方、NeuMF の推薦10件では **3.16%** にとどまった。

![履歴と推薦における低人気作品の割合](docs/popularity_bias.png)

履歴はユーザーごとの割合を平均した値。推薦では人気作品への偏りが強まっている。

## 2. Smooth xQuAD による調整

NeuMF の関連度上位100件から、履歴と推薦済みリストの構成を考慮して10件を選び直す。

1. 各ユーザーの履歴から、人気・低人気作品の比率を求める。
2. 履歴での比率に対して、推薦リストに十分含まれていないカテゴリへボーナスを付ける。
3. **元の関連度 ＋ λ × ボーナス** が高い作品を1件ずつ選ぶ。

再学習せず、ユーザーの履歴を考慮して低人気作品の推薦機会を増やす。
λ は val の HR・NDCG の低下をそれぞれ絶対値0.005以内に抑えた候補から、低人気作品の推薦枠が最大となる **0.8** を選択した。
手法は [Abdollahpouri et al., FLAIRS 2019](https://cdn.aaai.org/ocs/18199/18199-78818-1-PB.pdf) に基づく。

## 3. 改善結果

MovieLens 1M／6,040ユーザー／seed 42／推薦10件。同じ NeuMF に再ランキングを適用した。

![Smooth xQuAD の適用前後](docs/reranking_results.png)

低人気作品の推薦枠は **3.16% → 8.75%**、推薦された低人気作品数は **622 → 863作品** に増加した。
全体の HR@10 は **8.89% → 8.81%**、NDCG@10 は **0.04453 → 0.04404** とわずかに低下した。
精度の低下を抑えつつ、低人気作品への推薦機会を広げることができた。

## 課題・今後の展望

- **モデルのハイパーパラメータ探索**：埋め込み次元、層数、学習率、負例数を調整し、推薦精度を高める。
- **候補生成の改善**：現在の上位100件に入りにくい低人気作品も候補に含め、再ランキングで選べる範囲を広げる。
- **精度と推薦機会の両立**：候補数や補正の強さ、関連度スコアの調整を検討し、精度低下を抑えながら低人気作品の露出を増やす。

## 実行方法と構成

元データ `data/ml-1m.train.rating`・`data/ml-1m.test.negative` と、notebook の `run_dir` に指定した保存済みモデル・設定を配置し、上から実行する。
必要なモデル入力は `NeuMF-pre/checkpoint.pt` と `.hydra/config.yaml`。データと重みは Git 管理外。
図2枚を `docs/` に保存し、集計 CSV や推薦リストは別ファイルに保存しない。

- `ncf_reproduction/`：NCF の再現実装。
- [popularity_reranking/reranking.py](popularity_reranking/reranking.py)：Smooth xQuAD と係数選択。
- [notebooks/neumf_pre_full_catalog_story.ipynb](notebooks/neumf_pre_full_catalog_story.ipynb)：人気度の偏り → 手法 → 改善結果。
- `docs/`：README で使用する図。

```bash
uv sync --extra cu126
```
