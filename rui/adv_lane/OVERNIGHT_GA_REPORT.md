# 夜間 GA 自己改善 REPORT (2026-05-16)

OpenCode が早期停止/ハングで未記のため Claude が代筆。数値は全て
`ga_bench` (実 vanning_eval 検証) の出力に基づく。

## サマリ

| 指標 | baseline | 最終 | 差 |
|---|---|---|---|
| 平均コンテナ数 (16 inst) | 10.625 | **10.438** | -0.187 |
| 平均 dN (対 beam) | 1.188 | **1.0** | -0.188 (≈16% 圧縮) |
| 失格インスタンス | 0 | 0 | 不変 |
| bench wall (無競合) | 108.3s | 125.4s | +17s (天井1500s に対し桁違いに高速) |

スイート = `hard_instances/honban_cat31/` curated hard 12 + 標準 dataset 4
(case_balanced/weight_bound/volume_bound/small_many seed42)。
ベンチ GA 設定固定 generations=20 / pop=14 / seed=1234。

## `run_ga` に入った変更 (rui/algorithm_a.py, +70/-5)

1. **beam ヒューリスティック種付け**: 初期集団を純ランダム shuffle でなく
   `antagonist._ITEM_ORDERINGS` の 4 key_fn (weight/volume/footprint/
   longedge desc) で並べた個体で種付け。dN ギャップへの主因対策。
2. **有界 memetic 局所探索**: 組換え後に最空コンテナ item の再挿入 /
   first-improvement 2-swap (step 上限あり)。
3. **停滞リスタート + 適応的 mutation**: 8 世代非改善で集団刷新、
   停滞に応じ mutation rate を base 0.2 から漸増。

## 反復トラジェクトリ (progress.log 要約)

- 走#1 (01:44-02:11, 6 反復): #3 のみ実装、best 10.562。早期停止。
- 走#2 (02:14-03:28, ~15 反復): #1+#2 実装、**10.438 を複数回再現**
  (02:36 / 03:01 / 03:17 / 03:27)。確定値。
- 走#3 (03:30-09:57): provider 初期化でハング、イベント皆無=コスト0、
  6.5h 逸失。cancel 済。

## Job A: beam スケール参照解 (併走, 完了)

destination×ordering 並列 ladder で完走:

| items | beam 箱数 | wall | vanning_eval |
|---|---|---|---|
| 400 | 27 | 315s | pass/違反0 |
| 700 | 48 | 953s | pass/違反0 |
| 1100 | 75 | 1977s | pass/違反0 |
| 1582 | **107** | 3596s | pass/違反0 |

成果物: `runs/overnight_beam/{big_instance,layout_result,timing}_n*.json`。
serial 推定 ~9h (n=1582 単体) を並列で全 ladder ~1.9h に短縮。

## Claude 直接追加実験 (2 件, いずれも 10.438 を超えず)

| 実験 | 結果 | wall | 結論 |
|---|---|---|---|
| container-elimination memetic (最疎コンテナを他へ全量再配置→N-1) | 10.438 (不変) | 130s | 疎コンテナの item が COG±3000/重量24t/幾何で他へ移せず発火せず |
| 2-swap 強化 (max_steps 3→10, top-3 個体へ適用) | 10.438 (不変) | 298s (2.3x) | 順列レベル局所探索は種付け済の順序を超えられず、コストだけ増 |

**重要な知見**: dN=1.0 ギャップは「緩いコンテナ」でも「順列の質」でもなく、
**`pack_items` の貪欲単経路デコード自体が構造的に持つ差**。beam は
beam幅48×分岐の探索でこれを詰めるが、GA は permutation→greedy decode
である限り種付けで埋めた以上は縮まない。→ 研究トラック (真共進化/
学習 solver) では **decoder 自体を branching/学習化** するのが本丸。

## 残課題 / 次の一手

- dN<1.0 には decoder 強化が必須 (例: pack_items に上位2候補コンテナの
  浅い beam を入れる)。範囲外の構造変更なので研究トラック [[proj-vanning-layout]] へ。
- ベンチ stochastic ノイズ (10.438⇄10.562 振動)。elite 決定論評価で低減余地。
- 走#3 のハング: opencode provider 初期化固着。長時間走は heartbeat 監視必須。

## 成果物の所在

- 改善版: `rui/algorithm_a.py` (未コミット, branch `auto/rui-ga-overnight`)
- 保護スナップショット: `runs/overnight_ga/algorithm_a.best_0329.py`
- baseline: `runs/overnight_ga/baseline.json` / 軌跡: `progress.log`
