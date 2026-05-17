# 夜間自走タスク: GA (run_ga) の改善

あなたは作業者レーン。Claude が設計判断済みの **限定された範囲** を実装する。
独断で設計を広げないこと。一晩、下記プロトコルを反復せよ。

## 目的

`G:\マイドライブ\ideamap\worksp\vanning-algo\rui\algorithm_a.py` の
`run_ga` を改善し、固定ベンチスイート上で **平均コンテナ数 (mean_containers)**
と **平均 dN (対 beam)** を下げる。制約: 失格を増やさない、総 wall-time を
baseline の 3 倍以内に保つ (beam より十分速いまま)。

## 唯一の検証コマンド (cwd=vault root のまま叩ける ASCII パス)

```
python scripts/opencode/vanning_ga_bench.py bench      # ベンチ実行。最終行に VERDICT: ...
python scripts/opencode/vanning_ga_bench.py backup     # 現 algorithm_a.py を good 退避
python scripts/opencode/vanning_ga_bench.py restore    # good から algorithm_a.py 復元
```

- `bench` は 1〜2 分かかる。**途中で殺さず最後まで待て**。最終行に
  `VERDICT: IMPROVED|REGRESSED|NOCHANGE ...` が出る。これが唯一の真実。
- `cd` は permission deny。`git checkout` / `git restore` も deny。
  revert は必ず `restore` サブコマンドを使え。
- baseline.json は既に確定済み。`baseline` は叩く必要なし。

## 参照すべき既存 primitive (再利用せよ。再発明禁止)

- `rui/adv_lane/antagonist.py` の `_ITEM_ORDERINGS`
  = beam が強い 4 つの decreasing 順 (weight/volume/footprint/longedge desc)。
  各要素は `(name, key_fn)`。`key_fn` を `sorted(items, key=key_fn)` で使う。
- `rui/algorithm_a.py` の `pack_items`, `find_best_placement`,
  `candidate_score`, `evaluate_solution`, `fitness_key`, `order_crossover`,
  `mutate`, `Item`, `Container`。

## 許可技法 (これ以外の構造変更は禁止)

1. **初期集団のヒューリスティック種付け**: `run_ga` の初期 population を
   純ランダム shuffle でなく、`_ITEM_ORDERINGS` の 4 key_fn で
   `base_items` を並べたものを種として混ぜる (残りはランダム)。
2. **有界 memetic 局所探索**: 交叉/突然変異後の個体に軽い局所改善を入れる。
   例: 最も空い (fill 最小) コンテナの item を抜いて `find_best_placement`
   で再挿入を試す / 順列の first-improvement 2-swap。
   **必ずステップ上限**を設け、1 個体あたりの局所探索コストを定数で抑える。
3. 停滞リスタート / 適応的 mutation rate / tournament size・elitism 数の調整。

## 禁止

- 配置ジオメトリ (`can_place` / `overlaps` / `is_supported` /
  `generate_candidate_points` / 座標系) の再設計
- `vanning_eval`、`generate_items`、`ga_bench.py`、`beam_reference_scale.py`、
  `scripts/opencode/**` (ラッパー自身) の改変
- `git commit` / `git add` / `pip install` (どれも permission deny)
- pytest 等 `cd` を要する他コマンドの実行 (cwd 制約で deny される)

## ループ手順 (厳守)

1. 最初に 1 回 `backup` を実行 (good = 現状)。
2. `bench` を 1 回実行し現状の VERDICT/数値を確認 (起点把握)。
3. 以降ループ:
   a. `algorithm_a.py` に **1 つの焦点を絞った変更**を加える。
   b. `bench` を実行。
   c. 最終行が `VERDICT: IMPROVED` なら変更を採用 →
      ただちに `backup` で good を更新。
   d. `REGRESSED` または `NOCHANGE` なら `restore` で変更を捨て、
      別アプローチへ。
4. 停止条件: **5 回連続で IMPROVED が出ない**、または打つ手が尽きた、
   または bench が繰り返し失敗する → ループ終了。

## 報告 (必須)

- ベンチ結果は `ga_bench` が
  `worksp/vanning-algo/rui/adv_lane/runs/overnight_ga/progress.log`
  に毎回追記する (あなたは触らなくてよい)。
- ループ終了時、
  `worksp/vanning-algo/rui/adv_lane/runs/overnight_ga/REPORT.md`
  を新規作成し以下を書く:
  - before / after の mean_containers・mean_dN・disq・wall
  - 採用した変更点 (どの技法を入れたか) を箇条書き
  - 試して捨てた案と理由
  - 残課題・気になった点 1〜2 点

## 絶対則

**`ga_bench` の VERDICT 数値で IMPROVED を示せない限り「成功」「改善した」
と報告してはならない。** 数値の裏付けのない楽観報告は無価値。
最終的に good に残った `algorithm_a.py` が成果物 (commit はしない。
Claude が朝に `git diff` でレビューする)。
