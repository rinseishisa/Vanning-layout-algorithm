# rui adv_lane: 31種カタログ拡張 設計書（Phase 1.5 レーン B）

> Claude 設計成果物。実装はこの仕様に従う。上位: `design.md`（敵対レーン本体）/
> `rui/要件定義書.md`（生成規模・密度モデル根拠）/ ideamap `proj-vanning-layout` レーン2 B。
> 制約継承: `algorithm_a.py` / `generate_items.py` を**改変しない**（import 再利用のみ。design.md §5）。

## 0. 動機（A 検証結果からの接続）

A（regret shaping）の A/B（同 seed42/G8/pop8、`runs/step1_cma_g8p8` vs `step1_shaped_g8p8`）:

| 指標 | 整形なし | 整形あり |
|---|---|---|
| dN≥1 GA-hard 発掘 | 1 件 | **10 件** |
| best_r≥1 だった世代 | 1/8 | **6/8** |
| 全 eval mean regret | 0.036 | **0.185** |

shaping で「平坦で追尾不能」は解決。**ただし発掘された GA-hard は全て dN=1 ちょうど**（3種では
GA の失敗深さ天井が 1 コンテナ）。本レーン B の目的は **深さ天井を上げる**＝箱種多様性を増やし
組合せ摩擦を深くして dN≥2 の深い失敗を出す。A が「頻度・追尾性」を、B が「深さ」を担う相補設計。

## 1. 成功基準（受け入れ）

3種 shaped baseline（`runs/step1_shaped_g8p8`）比で、同条件 run にて:

1. **dN≥2 のインスタンスが出る**（深さ天井が 1 を超える）= B の主目的
2. dN≥1 発掘率が baseline 以上（領域が広がる、退化しない）
3. feasibility 健全維持: none率 baseline 同等以下、size-entropy が下限張付き連発しない
4. `vanning_eval --batch` で上位 hard instance が drop-in 互換（スキーマ不変）

未達時の撤退: design.md §4 撤退表に従い antagonist 1-step 先読み（②）へ戻る、または
greedy-variant regret 縮退。本レーン単独に固執しない。

## 2. 実カタログ仕様（実寸入手済 2026-05-15）

> [!success] 実寸データ入手・取込済
> いすゞロジスティクス開示資料 `(抜粋)ケースリスト.xlsx` を 2026-05-15 入手。
> **木箱10種(id 1-10)＋スチール21種(id 11-31)＝実 31 種**、寸法 mm、を抽出し
> `rui/adv_lane/catalog31.py`（純データ定数）に格納・検証済。当初想定の合成
> プレースホルダは**不要になった**（本節以下は実データ前提に改訂）。
> - 原典に**重量列なし** → weight は密度モデル（`rui/要件定義書 §2`）を維持（設計不変）。
> - 原典 名称列は日本語 xlsx encoding 罠で非可逆破損 → 非本質ゆえ ASCII ラベル合成
>   （材料/寸法/id は anchor 照合で検証済・正）。実名称が要るなら別リーダ（Excel COM 等）で再取得。
> - **座標写像**: `h = シート H 列`（天地固定の鉛直、不変）。`w,l = シート L,W 列`
>   （配置時 X-Y 90° 回転等価なので順不同）。

### 2.1 既知アンカー（不変）

| 区分 | W × L × H (mm) | 体積 m³ | 由来 |
|---|---|---|---|
| 小 | 760 × 1130 × 550 | 0.472 | team 要件定義書（実寸） |
| 中 | 1490 × 2260 × 900 | 3.030 | 同上 |
| 大 | 2280 × 2550 × 2355 | 13.690 | 同上 |
| コンテナ内寸 | 12000 × 2300 × 2400 | 66.24 | 同上（収容上界） |

**検証済**: Phase1 の既知3種は実31種の真部分集合だった —
小=case **#24**、中=case **#6, #16**、大=case **#5**（anchor 照合一致）。
よって 3種 baseline は 31種の部分集合で、**A の知見（shaping）が連続的に引き継がれる**。
体積レンジ実測 **0.472 – 13.692 m³**、各辺はコンテナ内寸（12000/2300/2400）以内。

### 2.2 カタログ実体（`rui/adv_lane/catalog31.py`、実データ）

- 構造: `CATALOG_31 = [(id, material, label, w, l, h), ...]` 31 行・純データ定数
  （関数/分岐なし）。`material ∈ {wood(10), steel(21)}`。
- 密度はカタログに**持たせない**（原典に重量列なし）。`rui/要件定義書 §2` の
  size 別密度モデルを material 別帯へ一般化: wood ρ∈[40,180]、steel ρ∈[120,600] kg/m³
  を **§3 の θ `rho_shift/rho_gain` で駆動**（VOLUME-BOUND 既定維持、weight×volume 緊張は knob）。
- 不変条件（pytest）: 31 行 / wood10・steel21 / 各辺 ≤ コンテナ内寸 /
  `clip(vol×ρ,100,12000)` 全 type 非退化 / anchor #24,#6/#16,#5 が既知3寸法と一致。

### 2.3 実ケースリスト取込（完了）

`(抜粋)ケースリスト.xlsx` → `catalog31.py` 抽出・検証済（2026-05-15）。当初の
「合成→実寸差替」手順は**実行完了**。今後実名称や重量が開示されても、`catalog31.py`
の純データ定数を上書きするだけで θ/generator/loop/regret は不変（カタログは値、構造固定）。

## 3. θ 再パラメータ化（**次元爆発の回避が核心**）

> [!danger] 素朴な 31 logits は CMA-ES を破綻させる
> 現 θ は 13-D（size3+ρc3+ρw3+dest3+scale1）。31種を「31 独立 logits ＋ 31×2 密度」
> にすると θ≈90-D超。CMA-ES は概ね 50-D 超で共分散推定が崩れ収束しない
> （[[feedback_ml_repo_trial_template]] の throughput 見積りと同様、事前にコスト評価必須）。
> → **構造化低次元パラメータ化**で θ を ≈20-D 以内に抑える。

### 3.1 採用案: 2軸連続フィールド上の混合分布（θ ≈ 18-D）

31種を「**素材軸 material∈{wood,steel}** × **サイズランク軸 rank∈[0,1]**（体積昇順正規化）」
の 2D 格子に配置。混合比は格子上のパラメトリック分布で表現:

| θ ブロック | 次元 | 役割 |
|---|---|---|
| `mat_logit` | 1 | wood/steel 比（sigmoid） |
| `rank_mean_w, rank_mean_s` | 2 | 各素材のサイズランク中心（sigmoid→[0,1]） |
| `rank_conc_w, rank_conc_s` | 2 | 同 集中度（小=偏り＝難所探索 / 大=均し） |
| `rho_shift` | 1 | 全カタログ密度帯の一様シフト（現実帯内 clip） |
| `rho_gain` | 1 | 密度帯の拡縮（weight×volume 緊張 knob） |
| `dest_logits` | 3 | 現行流用（目的地配分） |
| `s_scale_raw` | 1 | 現行流用（体積バジェット） |
| 予備（素材別 skew 等） | ~7 | 表現力不足時のみ解放（既定 0 固定） |

合計 **約 11–18 D**（CMA-ES 安全域）。各 type の選択確率 =
`P(material)·BetaPMF(rank; mean, conc)` を 31 格子点で離散化し正規化。
→ 31種を「どの素材の・どのサイズ帯を・どれだけ尖らせて混ぜるか」の少数 knob で操る。
mode collapse（1 type 乱発）は `rank_conc` 上限と下記 entropy guard で抑制。

### 3.2 制約の再スケール（重要）

- **size-entropy guard**: 最大エントロピーが `ln3≈1.10`→`ln31≈3.43` に変わる。
  `H_MIN` を**正規化エントロピー** `H/ln(K)` 基準へ移行し閾値 0.30 を維持
  （絶対値 0.30 のままだと31種では事実上無制約化）。`generator._check_feasibility` を
  正規化版へ（adv_lane 内で完結、generate_items.py 非改変）。
- 体積バジェット・est_item/cont∈[8,45]・dest 下限 P_MIN は現行ロジック流用
  （`avg_vol` がカタログ加重平均になるだけで式不変）。

## 4. generate_items.py 非改変での 31種注入

design.md §5「generate_items.py を改変しない」を厳守する。`generate_items.generate_items`
は module-global `ITEM_TYPES`(3種) 依存。選択肢と判断:

| 案 | 内容 | 判定 |
|---|---|---|
| (i) global 差替 | adv_lane で `ITEM_TYPES` を monkeypatch | ✗ 副作用大・テスト汚染 |
| (ii) kwarg 追加 | `generate_items(case, seed, item_types=...)` 後方互換追加 | △ 小改変だが §5 抵触 |
| **(iii) adv_lane 局所サンプラ** | カタログ駆動の最小生成関数を `generator.py` 内に持つ（密度モデル式は `rui/要件定義書 §2` を**式として**再現、`_volume_m3` 等は import 再利用） | ✅ §5 厳守・採用 |

採用 (iii): `build_dataset` は 3種時 `generate_items` 経由を維持（後方互換）、
31種モード時のみ局所サンプラ経路。出力スキーマ（`dataset_info`+`items[...]`）は完全同一に保ち
`vanning_eval.load_items` / `algorithm_a` drop-in を不変条件テストで保証。

> [!danger] size_type は厳格 enum（2026-05-15 実地検出・修正済）
> `vanning_eval/schema.py` `VALID_SIZE_TYPES={"small","medium","large"}` が
> items_input / layout_result の**両方**で size_type を hard enum 検証する。
> **31カタログラベル(`steel_22` 等)を size_type に入れてはならない**。
> `generator31._size_class(w,l,h)` で体積を既知3アンカー幾何中点境界
> (1.197 / 6.441 m³) で {small,medium,large} へバケットする。size_type は
> 配置・採点・幾何に不使用（dimensions が実寸を保持、catalog 箱は一意 dims で
> 同定可）＝**dN/regret 完全不変の cosmetic 修正**。教訓: 寛容な中間経路
> (`algorithm_a` の DataFrame 取込は enum 非検証) で drop-in を判定するな。

## 5. 検証計画

1. 単体: `catalog31.py` 不変条件 / θ デコード往復（encode∘decode≈id）/ 正規化 entropy guard 境界
2. スモーク `--catalog 31 --smoke`: crash 無し・regret 分散・none率健全・hard drop-in
3. **A/B 本走**: 3種 shaped baseline と同 seed/G/pop で 31種 shaped run、§1 成功基準を判定
4. 上位 hard を `algorithm_a` + `vanning_eval --batch` に通し GA 苦戦（dN≥2 含む）を実証レポート
5. **スキーマ受け入れは実バリダで**: `vanning_eval.schema.load_items` / `load_layout` を import して
   items_input・layout_result の両方を実際に通すまでが合格（`algorithm_a` 経路通過で代用不可）。
   → [[know-adversarial-instance-3dbpp]] 罠4 / [[feedback_dropin_validate_against_real_validator]]

## 6. スコープ外 / 連携 TODO

- 実ケースリスト入手 = **チーム/client 依頼事項**（提出物 spec ではなく敵対レーン精度向上目的と明記）
- 31種で貨物属性制約（上積み禁止・危険物隔離）は Phase 2 本体課題、本レーンは寸法/密度/混合のみ
- 実装は本設計承認後（A→B の B は本設計書まで。実装着手は別ステップ）

## 7. 関連

- `design.md` — 敵対レーン本体（regret/antagonist/CMA-ES、§9 で31種を Phase2 と明示）
- `rui/要件定義書.md` — 生成規模・密度モデル根拠（§2 VOLUME-BOUND 維持）
- ideamap [[know-敵対レーン入門]] / [[know-adversarial-instance-3dbpp]] / [[know-prediction-error-as-preference-signal]]
- [[feedback_ml_repo_trial_template]] — 新規手法は事前コスト見積り必須（θ 次元 = CMA-ES 収束コスト）
