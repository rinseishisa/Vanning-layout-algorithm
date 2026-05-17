# rui/adv_lane — 敵対的貨物生成レーン (Phase 1.5)

## 依存

```bash
pip install -r rui/adv_lane/requirements.txt
```

`cma` が無い場合は内製 (mu,lambda)-ES にフォールバックします（サンプル効率は落ちます）。

## 起動

### 通常実行

```bash
python -m rui.adv_lane.loop --gen 15 --pop 12 --ga-gen 10 --ga-pop 10
```

### スモークテスト（軽量・高速）

```bash
python -m rui.adv_lane.loop --smoke
```

`--smoke` は `G=3, pop=4, ga_gen=5, ga_pop=6` で動作確認を行います。

### 31種カタログモード（Phase 1.5 レーン B）

```bash
python -m rui.adv_lane.loop --catalog 31 --smoke
```

`--catalog 31` を指定すると 31-type catalog (`catalog31.py`) 経由で生成・探索します。
省略時は `--catalog 3`（従来の 3 種レーン）がデフォルトです。

## 出力

- `rui/adv_lane/runs/<timestamp>/`
  - `trajectory.csv` — 個体ごとの regret / protagonist / antagonist 結果
  - `gen_summary.csv` — 世代ごとの統計（None 率、size-entropy、regret 平均/最大/標準偏差）
  - `best_theta.json` — 最終世代の最良 θ
- `rui/adv_lane/hard_instances/hard_<rank>_<tag>.json`
  - regret 上位 20 件の items_input（`vanning_eval` / `algorithm_a` にそのまま投入可能）

## 設計書

`rui/adv_lane/design.md` が SSOT です。
