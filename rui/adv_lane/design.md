# rui adv_lane: 敵対的貨物生成 設計書（Phase 1.5）

> Claude 設計 → OpenCode 実装 → Claude レビュー の「設計」成果物。実装はこの仕様に従うこと。
> 上位知見: ideamap `know-adversarial-instance-3dbpp` / `rui/要件定義書.md`。protagonist は `rui/algorithm_a.py`。

## 0. 目的

「貨物データ生成器（generator）」と「レイアウト solver」を minimax/regret で共進化させ、
現行 GA solver が苦手とする **現実規模かつ難しい** 貨物インスタンスを自動発見する。
非対称構成: **solver は固定**（protagonist=現 GA、antagonist=ビームサーチ強版）、
**generator のみ CMA-ES で学習**。RL solver 化は卒研以降。

## 1. データフロー

```
θ (generatorパラメータ, R^k)
  └─ generator.build_dataset(θ, seed) ─→ items_input 形式 dict
        ├─ protagonist: algorithm_a.run_ga(items)        ─→ (dq_p, N_p, dev_p)
        └─ antagonist : beam_search_strong(items)         ─→ (dq_a, N_a, dev_a)
                                                            │
                              regret.compute(p, a) ─→ scalar r  (None=破棄)
  CMA-ES: maximize E[r]  (= minimize −r)。1世代 = pop 個の θ サンプル
  mode-collapse ガード / feasibility gate / 上位 hard instance 保存
```

## 2. regret 定義（辞書式採点 surrogate / 本設計の核心）

採点は要件定義書 §5.3 の辞書式（①コンテナ数 ②重心ズレ平均 ③処理時間）。
処理時間は generator 学習信号に不適なので **(コンテナ数, 重心ズレ平均) の 2 段**を使う。

各 solver の出力を `R = (dq: bool, N: int, dev: float)` とする
（`dq`=失格, `N`=使用コンテナ数, `dev`=合格コンテナの `mean(|Yg−6000|)`）。

```
def compute_regret(p: R, a: R, *, eps: float = 1e-4, dq_bonus: float = 1e3) -> float | None:
    # feasibility gate (PAIRED 中核)
    if a.dq:
        return None                      # antagonist も解けない＝退化/不可能 → 破棄
    if p.dq and not a.dq:
        return dq_bonus                  # 強 solver は解けるが GA 失格 → 最良 hard instance
    # 両者合格: 優先順位保存スカラー化
    dN   = p.N   - a.N                   # protagonist が余分に使ったコンテナ数
    dDev = p.dev - a.dev                 # 重心ズレ平均の劣化
    return dN + eps * dDev
```

- **eps 根拠**: 合格時 `dev ∈ [0, 3000)`（3000mm 超は dq）。`|dDev| < 3000`。
  `eps=1e-4` なら `eps·|dDev| < 0.3 < 1 ≤ |dN|`（dN は整数差）→ **1 コンテナ差が常に dev 差を支配**＝辞書式優先を厳密保存。
- generator は `r` を最大化（`dN` 増 = GA をより困らせる方向）。
- `r is None` のサンプルは CMA-ES 評価から除外し再サンプル（mode-collapse 対策と一体）。
- `r < 0`（GA が beam より良い）も有効値として残す（負 regret = その θ 領域は非有望、CMA-ES が自然に避ける）。dq_bonus は他項より十分大きく hard instance を最優先で引き寄せる。

## 3. generator パラメータ空間 θ

`rui/generate_items.py` の現実的密度モデル（size 別 `clip(vol×ρ,100,12000)`）を土台に、
連続パラメータ θ ∈ R^k を CMA-ES で最適化。**現実性アンカーを hard 制約**で噛ませる。

| 要素 | パラメータ化 | 制約 |
|---|---|---|
| size 混合比 π (small/med/large) | logits 3 → softmax | **entropy ≥ H_min**（小箱乱発=mode collapse 防止、AR2L 知見） |
| size 別 密度中心 ρ_c[s] | sigmoid→`[ρ_lo[s], ρ_hi[s]]` 線形写像 | 現実帯（generate_items の case 帯を内包する広めの固定境界） |
| size 別 密度幅 ρ_w[s] | sigmoid→`[0, (ρ_hi−ρ_lo)/2]` | 重量は最終 `clip(vol×ρ,100,12000)` |
| 目的地配分 (D=3) | logits D → softmax | 各 dest 比率 ≥ p_min（極端な 1 dest 寡占を防止） |
| 規模 s_scale | sigmoid→`[0.6, 1.0]` | 総体積 ≒ s_scale × 0.8 × コンテナ容積 × n_cont_target。**item 数下限 ≥ 8×D**（1コンテナ最小ケース×dest 数） |

- item 生成: 総体積バジェットに達するまで π/ρ/dest からサンプリング（`generate_items.py` のロジックを関数として再利用、θ で各分布を駆動）。
- 出力は **items_input 形式 dict**（`dataset_info` + `items[{item_id,size_type,dimensions,weight,destination_id}]`）。`vanning_eval` / `algorithm_a` にそのまま投入可能。
- `n_cont_target`: 1 インスタンス ≒ 数コンテナ規模（dest 数×2〜3 本程度。現実規模 8–40 ケース/本を逸脱しないこと、`generate_items.sanity` の est_item/cont∈[8,40] 制約を流用検証）。

## 4. antagonist: ビームサーチ強版

protagonist（GA）より**徹底的に探索する決定的サーチ**。`algorithm_a.py` の配置プリミティブを再利用:
`build_items` / `rotated_dims` / `can_place` / `generate_candidate_points` / `find_best_placement` /
`candidate_score` / `y_deviation` / `Container` / `make_placed_item` を import して使う（再実装しない）。

アルゴリズム（**destination 群ごとに独立**、目的地混載は絶対禁止）:

```
beam_search_strong(items, beam_width=B, branch=K) -> List[Container]:
  for each dest group (algorithm_a と同じ分割):
    states = [ empty ]                     # state = List[Container]
    残り item 列 = build_items 準拠ソート（重い/大きい順）
    for item in 残り:
      cand_states = []
      for st in states:
        # 既存各コンテナ + 新規1コンテナ について find_best_placement 相当で
        # 上位 K 配置（rotation×candidate point を candidate_score でソートした上位）を展開
        for placement in top-K feasible placements(st, item):
          cand_states.append(apply(st, placement))
      # 部分状態を辞書式 surrogate で評価し beam_width に剪定
      states = nsmallest(beam_width, cand_states, key=partial_lex_key)
    best = min(states, key=final_lex_key)
  return 連結(best over dest groups)

partial_lex_key(st) = (使用コンテナ数, 進行中 dev 推定, デッドスペース)   # 小さいほど良い
final_lex_key(st)   = (使用コンテナ数, mean|Yg−6000|)
```

- B, K は実装時チューニング（初期 B=16, K=4）。**beam は GA より強い**ことが前提（同等以下のコンテナ数を出す）。
  もし多くのインスタンスで beam < GA でない（regret≤0 多発）なら B/K を上げる。撤退ライン: B=64,K=8 でも信号出ずなら antagonist を greedy variant 差（best-fit vs first-fit）に縮退（proj 撤退表）。
- antagonist が失格（どの dest 群も全 item を ±3000/重量内に収められない）→ `dq_a=True` → regret gate で破棄。

## 5. CMA-ES ループ

- ライブラリ: `cma`（pip）。**追加依存**として `rui/adv_lane/requirements.txt` に明記しインストール手順を README へ。`cma` 不在時は内製 (μ,λ)-ES にフォールバック（実装簡素版でも可、ただし cma 優先）。
- 1 世代: pop 個の θ → 各 θ で `build_dataset → protagonist ∥ antagonist → regret`。
  `regret is None` のサンプルは fitness を最劣（または再サンプル）にして mode-collapse を回避。
- 反復: 既定 世代数 G（初期 G=15, pop=12。スモークは G=3,pop=4）。
- **mode-collapse モニタ**（毎世代ログ）: 平均 size-entropy / item 数分布 / `None率`（infeasible率）/ regret 統計。
  None率 > 60% or entropy が下限張付き連発 → 警告（生成器が退化方向）。
- **POET-lite（stretch）**: 上位 hard instance を replay buffer（N=8）に保持、過去 instance での GA 劣化も監視（過適合検知）。時間なければ省略可。

## 6. 出力

- `rui/adv_lane/hard_instances/hard_<rank>_<tag>.json`: regret 上位 K（既定 20）の items_input。drop-in 互換。
- `rui/adv_lane/runs/<timestamp>/`: regret 軌跡 csv、世代ログ、最終 θ、モニタ図（matplotlib 任意）。
- 仕上げ: 上位 hard instance を `algorithm_a` と `vanning_eval --batch` に通し、**GA が実際に苦戦（コンテナ数増 or 失格）すること**を検証レポート化。

## 7. モジュール構成（rui/adv_lane/）

| ファイル | 役割 |
|---|---|
| `design.md` | 本書 |
| `regret.py` | §2 の `compute_regret` + `SolverResult` dataclass |
| `generator.py` | §3 θ encode/decode + `build_dataset(θ, seed)`（generate_items ロジック再利用） |
| `antagonist.py` | §4 `beam_search_strong`（algorithm_a プリミティブ再利用） |
| `loop.py` | §5 CMA-ES 駆動・モニタ・§6 出力。CLI: `python -m rui.adv_lane.loop --gen G --pop P [--smoke]` |
| `requirements.txt` | `cma`（理由コメント付き） |
| `README.md` | 起動・依存・出力の運用メモ |

## 8. 受け入れ基準（Claude レビュー観点）

1. `compute_regret` が §2 の gate/eps 仕様通り（境界: 両合格 / p失格 / a失格 / dN符号）。単体テストあり
2. `build_dataset` 出力が items_input スキーマ準拠（`vanning_eval.load_items` が通る）、現実性アンカー（entropy/density/dest/item数下限）を hard 充足、est_item/cont∈[8,40] を概ね満たす
3. `beam_search_strong` が algorithm_a プリミティブ再利用（配置判定の再実装なし）、destination 分割厳守、スモークで GA 以下のコンテナ数を多くのインスタンスで達成
4. スモーク（G=3,pop=4）が crash なし・regret に分散（定数でない）・None率が極端でない・hard instance が drop-in 検証で「GA 苦戦」を示す
5. 制約: `rui/` 配下のみ、commit/push しない、`algorithm_a.py`/`generate_items.py` を改変しない（import 再利用のみ）

## 9. スコープ外（本 Phase で作らない）

- RL solver 化（AR2L 完全再現）= 卒研以降
- neural generator（REINFORCE）= CMA-ES で完成扱い、時間あれば stretch
- 31 種フルグリッド = まず 3 種簡略のまま。31 種は Phase 2
