# design_coevo.md — 研究トラック「beam を超える (dN<0) 学習デコーダ」設計書

> SSOT: ideamap `proj-vanning-layout.md` レーン3 / `know-adversarial-instance-3dbpp.md` レーン3実証ログ。
> 本書は**作業現場の設計 SSOT**（実装判断・実験プロトコルを抱える）。ideamap 側は状態+次の一手のみ。
> 起票 2026-05-18（survey 完了・AB並行をユーザー決定）。

## 0. 位置づけ・スコープ

- **目的**: 学習デコーダで eval 参照 beam を **越える**（dN<0）。授業トラック（レーン1+2）の主納品とは独立。
- **方針**: 卒研化しない週末 hands-on prototype。Colab 前提（RL rollout で GPU が効く局面のみ）。
- **2026-05-18 決定**: Workstream **A（強教師 BC スパイク）と B（Expert Iteration）を並行**。C/D はエスカレーション/hedge として記録。
- **再利用資産**: レーン2/R0/R1 の実装一式（§6 マップ）。新規インフラは最小化。

## 1. 中心の組み替え — 「超-beam 信号源はどこから来るか」

これまでの全軌跡（greedy 1.33 → R0 固定beam 1.0 → R1-logreg 2.33 → R1-MLP 0.833 → 強DAgger **0.5**(監査値) → portfolio **0.31**）は **すべて eval beam の模倣 or 模倣の合成**。

> **模倣（BC/DAgger/portfolio）は構造的に eval beam が天井。** 強DAgger 監査値 0.5（ceiling 0.0 未到達）はこれを実証した。dN<0 は「eval beam をコピーする」系列の延長線上に原理的に存在しない。

⇒ 手法選定の唯一の軸は **「beam を越える信号をどこから注入するか」**。これを満たす源は4つしかない：

| # | 信号源 | dN<0 になり得る理由 | 文献 |
|---|---|---|---|
| **A** | eval beam **より強い教師**（広beam+2-swap後改善 / 多順序 keep-best / 小dest群は CP-SAT 厳密）→ 既存 BC/DAgger パイプライン | 天井そのものを「eval beam」→「より強い解」に差し替え。模倣の上限が eval beam を上回る | — (自家製、`rui/exact/` 既存) |
| **B** | **Expert Iteration**（学習方策→探索誘導→勝ち軌跡採択→再学習→反復） | 探索（improvement operator）が毎反復で方策を固定heuristic beam 超へ持ち上げ、不動点が beam を越える | Laterre+ 2018 *Ranked Reward* (arXiv:1807.01672, **2D/3D-BPP で MCTS/heuristic/IP solver 超を実証**) / Grill+ 2020 MCTS-as-RegPolOpt (arXiv:2007.12509) |
| **C** | **真の目的で RL**（PPO/POMO + regret整形報酬）。beam を模倣せず箱数を直接最適化 | 報酬が実目的＝beam の選択に縛られない | Zhao+ ICLR2022 (arXiv:2006.14978) / AR2L NeurIPS2023 / Kwon+ POMO 2021 / Pan+ 2025 Preference-Opt-for-CO (arXiv:2505.08735, **辞書式/疎報酬に直撃**) |
| **D** | **学習 improvement operator**（Neural LNS / 学習 destroy-repair / neural-SA） | 構築の天井と独立に解空間で改善＝構築heuristic を原理的に越える | Hottung&Tierney NLNS / Zhang+ 2022 DRL-guided improvement (arXiv:2211.10936) / Correia+ 2022 Neural-SA (arXiv:2203.02201, BPP含む) |

**A/B を主、C を escalation、D を hedge** とする（§4-5）。

## 2. 物差しの定義と落とし穴（D-2 前提崩壊の教訓を内蔵）

- **eval 参照**: `ga_bench.build_beam_ref` → `antagonist.beam_search_strong`（**既定 48/12**, dest 再グループ化 + `_ITEM_ORDERINGS` 多順序 keep-best, 勝ち判定 `_final_lex_key`）。dN = `ga_containers − beam_ref_N`。
- **🔴 発見（2026-05-18）**: R1 教師 `r1_teacher.py` は **24/6**（`TEACHER_BEAM_WIDTH=24 / TEACHER_BRANCH=6`）＝ **eval 参照 48/12 より弱い beam を模倣していた**。「模倣が eval beam を越えない」のは当然で、A の最短レバーはここ。
- **自己参照の罠（certified-LB dead-end `8a0eb75` の再来防止）**: 「幅4には勝つが幅64には負ける」は beam 超えではない。**dN<0 主張時は必ず (i) 固定参照=48/12 `beam_search_strong` (ii) 強 wide-beam（例 96/24 + 2-swap）にも併記 (iii) `rui/exact/per_destination_lb` で certified 下界 sanity** の3点をセットで報告する。下界が縮退する dest 群（26-54個, `8a0eb75` で実証）は (iii) を non-degenerate な小 dest 群限定で使う。

## 3. Workstream A — 強教師 BC スパイク（最安・最初・de-risk）

**仮説**: dN が下がらないのは「天井=eval beam」だからでなく「教師<eval beam だった」から。教師を eval beam より強くすれば純 offline BC で dN<0 が出る。

**1ノブ差し替え**（`r1_teacher.py` を改変、`r1_train`/`ga_bench` は不変）:

1. **教師 beam 強化**: `TEACHER_BEAM_WIDTH 24→≥48`, `TEACHER_BRANCH 6→≥12`（= eval 参照と同等以上）。`extract_instance` の各 dest 群 winner に **2-swap 後改善**（`algorithm_a.local_search_2swap` 相当）を1段かける。
2. **小 dest 群は CP-SAT 厳密教師**: dest 群サイズが厳密可能域なら `rui/exact/cpsat_model` で最適配置を教師に（beam でなく証明付き最適を模倣）。閾値は §3 Step-0 で決定。
3. 再学習 `r1_train --model mlp` → 評価 `ga_bench --decoder learned --offset/--limit`（Fold A/B 両方）。

**Step-0 規律（必須, [[feedback_ml_repo_trial_template]] / R0 timeout 反省）**: 本走前に強化 beam での `r1_teacher._step0()` 改で **抽出 wall/inst・全16inst推定・fit時間** を実測。48/12 は 24/6 比で抽出が重い → 推定が天井（数時間）を超えるなら CP-SAT 教師主体に切替 or beam 幅を二分探索。

**撤退ライン**: 強教師(eval beam同等以上)でも held-out dN が portfolio 0.31 を下回らない → 「offline BC の天井は教師強度でなく pointwise greedy decode の構造（R1診断 (b) lookahead 喪失）」と確定し、B に重心を移す（A は floor 更新の成果として記録）。

**期待**: dN<0 への最短路。新インフラ・GPU・Colab 不要、純 CPU。B の設計を確定する前に「天井は本当に beam か」を実証で潰す。

## 4. Workstream B — Expert Iteration（本命の研究レバー）

**核**: 学習スコアラが**探索を誘導**し、探索が固定heuristic beam を越える解を産み、それを教師に巻き戻して再学習。不動点が eval beam を越える（Ranked Reward の bin-packing 実証と同型）。

### MDP 定義（既存 featurize と整合）

- **state**: `List[Container]` 部分配置 + 残 item 列（`r1_teacher.featurize` の15次元がそのまま観測; 必要なら拡張）。
- **action**: 次 item の配置候補 = `_top_k_placements(state, item, k)` の中から1つ（離散, 可変長, feasibility は primitive が保証）。
- **reward**: 辞書式 regret 整合。終端 `−(_final_lex_key)` 由来。**疎整数 dN プラトー対策にレーン2実証済の regret shaping を流用**（最空コンテナ空き率 `1−min_fill` を連続 slack proxy、λ<1 で辞書式順位は保存; know罠1解決と同型）。
- **policy/value**: R1-MLP スコアラを policy prior に流用。value は省略開始（Ranked Reward は value-free でも回る）。

### Expert Iteration ループ

```
init: π0 = R1-MLP（A の強教師版があればそれ）
repeat:
  1. π_t を beam/MCTS の枝刈り・分岐スコアに使い (coevo_decoder.pack_items_beam の
     _partial_lex_key を π_t proba に置換) 全 train inst を探索デコード
  2. Ranked Reward 採択: 各 inst の探索解のうち「現 best を _final_lex_key で
     上回ったもの」だけ採択 (閾値 = 過去 rollout 分布の上位分位; Laterre R2 方式)
  3. 採択軌跡を r1_teacher.featurize で trace 化 → base BC に集約
  4. π_{t+1} = MLP 再訓練 (r1_train パイプライン再利用)
  5. ga_bench --decoder learned で dN 計測、§2 三点セットで監査
until dN 改善停滞 or 絶対時間天井
```

- **推論形態**: 学習後の本番デコードは **learned-greedy（探索不要 = greedy 速度, `pack_items_learned`）**。探索コストは訓練時のみ（R0 の ~6x は train-time に隔離）。
- **Colab**: ループ自体は小幅 beam なら CPU 安。net を MLP→小 Transformer/PCT に拡大 or MCTS 深化する段で GPU が効く。まず CPU で不動点が eval beam を越えるかを実証してから Colab スケール。
- **撤退ライン**: 3反復で探索採択率が単調減 or dN が A floor を割らない → MCTS 化（Grill 2020 reg-pol-opt）or C へ。

## 5. C / D（escalation・hedge, 設計は起動時に詳細化）

- **C: RL 方策**（A+B 頭打ち時）。POMO 型 multi-rollout REINFORCE（critic 不要・低分散）+ regret 整形 + 必要なら Preference-Opt（順序化で疎整数報酬の分散を吸収）。Zhao ICLR2022 の feasibility-mask actor-critic がアーキ雛形（R0 で 45分 arch-read 済の系）。**Colab GPU が最も効く・最重量・高分散**。AR2L の robust-RL-solver 半分＝generator レーンとの真共進化はこの上に乗る。
- **D: Neural LNS / 学習 improvement**（並走 hedge）。`local_search_2swap` を学習 destroy-repair 方策に。構築の天井と独立。A/B が構築側で詰まっても解空間側で dN を稼げる保険。中量・Colab 任意。

## 6. 再利用マップ（実ファイル → 役割）

| ファイル | 役割 | A | B |
|---|---|---|---|
| `r1_teacher.py` | 教師 beam trace 抽出 + `featurize`(15D) | **改変点（教師強化）** | trace 化に流用 |
| `r1_train.py` | MLP 学習 (`MODEL_PATH`) | 再学習 | 各反復で再学習 |
| `coevo_decoder.py` | `pack_items_beam`(R0) / `pack_items_learned`(R1推論) | learned 評価 | **beam 誘導の枝刈りキーを π に置換** |
| `r1_dagger.py` | 強 oracle (`--strong`, full beam-completion) | 強教師の参考実装 | oracle 強度規律の前例 |
| `ga_bench.py` | 実 vanning_eval 駆動 dN 計測 (`--decoder/--offset/--limit`) | 監査 | 各反復監査 |
| `antagonist.py` | `beam_search_strong`(48/12, eval参照) / `_top_k_placements` / `_apply_placement` / `_final_lex_key` | 物差し | action/遷移 primitive |
| `rui/exact/` | `per_destination_lb`(certified sanity) / `cpsat_model`(小群厳密教師) | **CP-SAT 教師源** | dN<0 の certified sanity |
| `loop.py` | regret shaping（最空コンテナ slack proxy） | — | **reward shaping 流用** |

## 7. 決定ログ / 未決

- 2026-05-18: AB並行をユーザー決定。survey 本体は §1 表 + 文献。
- 2026-05-18: 物差し＝48/12 `beam_search_strong` 固定 + 強wide-beam併記 + certified LB sanity の三点（自己参照罠 `8a0eb75` 教訓内蔵）。
- 2026-05-18: **A スパイク（幅のみ強化）2-fold → fold 不安定**（§8 所見1-6）。幅でなく質。
- 2026-05-18: **A §3（48/12 + 2-swap 質的強化）2-fold → robust dN→0 確定**（§8 所見7-10）。Fold A 0.0(全件 beam tie)/Fold B 0.333、両 fold 改善・退行ゼロ・portfolio 0.31 超。**§3 中核仮説（質≫幅）確定・A スパイク完了**。但し純BCは beam を*越え*ない（tie が上限）と実証＝dN<0 は B 依存。
- 2026-05-18: **B スパイク（ExpIt 骨格スモーク）→ 機構正・naive 過適合**（§9）。未決②（quantile-α 等）は必須と確定。
- 2026-05-18: **B→A 両実装完了（ユーザー決定 1セッション）**。`coevo_expit.py` 新規 / `r1_teacher.py` 2-swap env opt-in。次＝B 未決②実装→本走（dN<0 本命）／A は CP-SAT 教師（未決①）で dN<0 追撃 任意。
- 2026-05-18: **B 未決② 実装完了**（`coevo_expit.py` 全面改修）: (a) Ranked-Reward 分位 α 採択（履歴 α 分位以上, `--alpha 1.0`=legacy strict）(b) 内部 val model-selection + patience 早期停止（fold hold-out/trace 収集と別）(c) base:ExpIt 行数バランス（seed 付 subsample, MLP は sample_weight 非対応ゆえ行数制御 + 有界 replay buffer で無限累積排除）(d) MLP L2 alpha + early_stopping、`--seed` で run1==run2。smoke（2-iter）で4機構動作 + **選択ガードが退行 iter2 を正しく不採用**（naive smoke の 1.167 過適合の防御が機能）を確認。本走ドライバ `b_honban.py`（2-fold + scorer.joblib backup/restore + 機械可読 summary）。
- 2026-05-18: **B 未決② 本走 2-fold = dN<0 未達・A §3 floor 下回り**（**同設定再走防止の記録**）。Fold A mean_dN **0.833**（選択=π0; ExpIt のどの反復も内部 val で π0 を超えられず early-stop）/ Fold B **0.5**（選択=it3; 内部 val 11.333→11.000 改善も hold-out は beam 越えず）。2-fold avg 0.667・below0 **0/12**・disq0。A §3 robust floor（0.0/0.333, avg≈0.167）を**両 fold で下回る**。**主交絡**: `--bootstrap-pi0` が val 切出後 **7/10 inst** で π0 再訓練 ＝ A §3（全10 inst）より弱い π0 から開始（Fold A は弱化 π0 すら回復できず, 選択=π0）。certified-LB sanity（§2 iii）は頭余地存在を確認（Fold A gap 2-3 / Fold B hard_11,12 gap 3-4, 但し Fold B std 3件は gap 0=beam 証明上最適で dN<0 構造的に不可）＝失敗は noise でなく手法。詳細 §11。
- 2026-05-18: **B-clean（π0=A §3 フル固定で交絡除去）完了＝B/ExpIt 路は dN<0 として exhausted・但し記録上最良 floor**（§12）。Fold A 0.0 / Fold B 0.167（2-fold avg **0.083**, A §3 純BC ~0.167・bootstrap 0.667 を上回る）。Fold B で ExpIt が hard_11/12 を beam-tie まで押下げ 0.333→0.167。**但し below-beam 0/12**＝**A §3 純BC・B bootstrap・B-clean の 3-config すべてで beam を厳密に下回らず**（certified-LB で頭余地は存在）＝design §1「同 48/12 beam 族の模倣/ExpIt は beam-tie が構造天井」を 3-config で**確証**。∴ **dN<0 は eval beam より厳密に強い信号源が必須**＝未決①（CP-SAT 厳密教師, 最安）or C（真目的 RL）へエスカレーション。floor を上げる工学と ceiling を破る研究は別物と実証的に確定。
- 2026-05-19: **D-2 = (b) beam 基準で受容に確定（§14）**。未決③ クローズ。授業トラック締めの一環（ユーザー判断）。
- 未決①: A の CP-SAT 教師＝`cpsat_model` は count のみ返す → solver 解抽出（cont/x/y/z/rot）+ trace-replay 実装が必要。dest 群サイズ閾値も Step-0 で。
- 未決②: B の Ranked-Reward 分位 α + held-out model selection + base/ExpIt 行数バランス + 正則化（スモークで必須と実証）。探索器（beam/浅MCTS）も。
- ~~未決③~~ **resolved (2026-05-19) → §14**: D-2（敵対生成器の regret 目的を beam 相対→下界ギャップ）は `8a0eb75` で前提崩壊 → **(b) beam 基準＋限界明記で受容**に確定（ユーザー判断「授業トラックを締める」）。詳細・限界・波及は §14。

## 8. 実証ログ — A スパイク（教師 beam 幅強化）

**プロトコル**: `r1_teacher.py` に env オーバライド追加（既定24/6不変＝後方互換, r1_dagger 非影響）。教師 beam を `TEACHER_BEAM_WIDTH/TEACHER_BRANCH` で段階強化し、`r1_train --model mlp` で再学習 → `ga_bench --decoder learned --offset/--limit` で Fold A(hold-out hard_01..06) / Fold B(hold-out suite[10:16]) の dN を実 vanning_eval 計測。物差し=既存 `beam_ref.json`（48/12 `beam_search_strong`）。

| 教師 | 抽出wall/inst | Fold A dN | Fold B dN | per-inst artifact |
|---|---|---|---|---|
| 24/6 (R1-MLP, 既存記録) | 16.2s | 0.833 | 0.5(監査値) | (know レーン3ログ) |
| 48/12 (=eval参照同強度, 幅のみ) | 32.3s | 0.667 | — | `runs/overnight_ga/spikeA_foldA_teacher48x12.json` |
| 96/24 (幅のみ・eval参照超) | 51.6s | **0.333** | **0.667**(退行) | `spikeA_foldA_teacher96x24.json` / `spikeA_foldB_teacher96x24.json` |
| **48/12 + 2-swap (§3 質的強化)** | **62.1s** | **0.0** | **0.333** | `spikeA3_foldA_teacher48x12_2swap.json` / `spikeA3_foldB_teacher48x12_2swap.json` |

**監査済所見（fold毎 single-run、保守側に倒した解釈）**:

1. **Fold A は教師強度に単調**（0.833→0.667→0.333, hard_01 +2→+1）。teacher 強度は実レバー。
2. **だが Fold B は逆**: 96/24 で 0.5(R1-MLP監査値)→0.667 ＝ **greedy 水準（Fold B greedy=0.667）へ退行**。同じ「より強い教師」が A 改善・B 退行 ＝ **教師 beam 幅の強化は fold 不安定で robust な dN<0 レバーではない**。
3. 2-fold 平均 ≈0.5 ＝ 強DAgger監査値と同水準、portfolio 0.31 未達、dN<0 は遠い。
4. **教訓（再挑戦防止）**: 「teacher を広い beam にするだけ」は単一 fold では劇的に見えるが汎化しない（R1診断(b) lookahead 喪失が fold 毎に不均一に効く）。Fold A だけで headline を書くのは強DAgger 0.0 誇大と同型の罠。**A は §3 完全仕様（質的に強い教師）で、かつ 2-fold ゲート必須**。
5. **certified-LB sanity（hard_01..06）**: perdest_LB=7–8（非縮退、`8a0eb75` の自明値1とは別）。eval beam 自身が証明上 2–3 箱非最適 ＝ dN<0 は noise でなく実頭余地あり。但し下界は tighten 不可（`8a0eb75`）＝物差しは「48/12 beam 比 dN」主、LB は頭余地の存在証明として併記。
6. **幅のみ強化の結論**: 純 offline BC を「広い beam 教師」だけで強化は fold 不安定（96/24: A 改善/B greedy退行）。robust でない。

**A §3（質的強化 = 48/12 + 2-swap 後改善教師）2-fold 結果（2026-05-18, single-run/fold）**:

7. **2-fold robust に dN→0**: Fold A **0.0**（hold-out hard_01..06 全6件 beam と完全 tie, hard_01 +2→0）/ Fold B **0.333**（hard_12,small_many のみ +1）。**両 fold で改善・退行ゼロ・disq0・greedy 速度**。2-fold 平均 ≈0.167。
8. **「質 ≫ 幅」を分離実証**: 48/12+2swap (Fold A 0.0) は 96/24 幅のみ (0.333) を凌駕、しかも 96/24 が fold 不安定なのに 2-swap は両 fold 改善。**§3 中核仮説（レバーは教師の質、beam 幅でない）を 2-fold ゲート通過で確定**。trace-replay 不要設計（2-swap は順列ジェノタイプ精製＝精製順列の beam decode 自体が正当 trace）が効いた。`local_search_2swap(best_ordered, max_steps=2, decoder=pack_items_beam)`、env opt-in `TEACHER_2SWAP_STEPS`（既定0=後方互換）、cost 抑制で 4順序 keep-best の best 1本のみ精製。
9. **但し dN<0 未達**: 純 offline BC + 2-swap 教師は beam 天井に **robust に到達するが越えない**（Fold A=0 tie / Fold B=+0.333）。教師が eval beam と同 beam 族（48/12）の 2-swap 精製ゆえ。**dN<0 には (i) 教師を eval beam より厳密に強く＝CP-SAT 小群厳密[未決①, cpsat_model は count のみ返す→solver 解抽出+trace-replay が必要] or 2-swap step↑/base beam↑ (ii) B の推論時 lookahead**。
10. **certified-LB sanity（hard_01..06）**: perdest_LB=7–8（非縮退）。eval beam 自身が証明上 2–3 箱非最適＝dN<0 は noise でなく実頭余地あり（下界 tighten 不可 `8a0eb75` ＝物差しは 48/12 beam 比 dN 主, LB は頭余地存在証明として併記）。

## 9. 実証ログ — B スパイク（Expert Iteration / Ranked-Reward 骨格スモーク）

**実装**: `coevo_expit.py`。`_traced_beam_for_group` ミラーで枝刈りキーを `_partial_lex_key` → **累積方策 logprob** に置換（policy-guided beam）。strict-improvement 採択（その inst の現 best を `_final_lex_key` で上回った勝ち軌跡のみ新教師）→ base BC に累積集約 → MLP 再学習 → 反復。推論は learned-greedy（探索は train-time のみ）。Step-0: policy-guided beam 19.6s/inst。

**スモーク結果（3反復, Fold A 規約, π0=当時 scorer.joblib）**:

1. **機構は正しい**: improved 0→**6**→**4**（iter1 全初観測=0 想定通り, iter2 で π_1 誘導 beam が train 10inst 中6件 strict 改善, iter3 で4件）＝**単調自己改善ループが生存・collapse 無**。骨格 end-to-end 成功（base 4774→agg 14519 rows）。
2. **だが naive 版は held-out 過適合**: Fold A dN **1.167**（greedy 1.33 をわずかに上回るのみ, hard_01 +3 最悪）＝**退行**。strict改善のみ＋累積集約が train 特異軌跡を memorize し base BC を希釈、汎化崩壊。
3. **決定的含意**: design_coevo.md 未決②（**Ranked-Reward の quantile-α + held-out model selection + base/ExpIt 行数バランス + 正則化**）は optional でなく**必須**。strict-improvement（α→max）は最も過適合しやすい設定で smoke が露見（spike が時間溶解前に真の設計要件を確定＝プロジェクト規律通り）。B 本体は未決②を組み込んでから本走。

## 10. 現時点の AB 配分（2026-05-18 両スパイク後）

- **A §3 = robust な「beam tie」floor を確立**（2-fold: 0.0 / 0.333、退行ゼロ、CPU・no-Colab・greedy速度）。portfolio 0.31・強DAgger 0.5 を上回る現最良の安定 floor。次の A 増分＝dN<0 へ: CP-SAT 小群厳密教師（未決①、solver 解抽出+trace-replay 要）or 2-swap step↑。
- **B = dN<0 の唯一の原理的路だが naive 版は過適合**。骨格は機構実証済、未決②（quantile-α 等）を実装してから本走。**純 BC（A）は beam を*越え*られない（tie が上限）ことを A §3 が実証した以上、dN<0 は B に懸かる**。
- 結論（両スパイク後）: A §3 を robust floor として確定・記録、B を未決②実装後に本命継続。
- **更新（B 未決② 本走後, 2026-05-18）**: bootstrap 版は **dN<0 未達・A §3 floor 下回り**（§11）だが `--bootstrap-pi0`×小プールで π0 弱化の交絡。
- **確定（B-clean 完了, 2026-05-18）**: π0=A §3 フル固定で交絡除去（§12）。**Fold A 0.0 / Fold B 0.167（2-fold avg 0.083）＝記録上最良の robust floor**（A §3 純BC ~0.167 / bootstrap 0.667 を上回る）。Fold B は ExpIt が hard_11/12 を beam-tie まで押下げ 0.333→0.167。**但し dN<0 は 0/12 で未達**＝ExpIt は floor を beam-tie へ robust に押し上げるが **beam ceiling は破れない**。A §3 純BC・B bootstrap・B-clean の **3 構成すべてで below-beam 0/N**（certified-LB で頭余地は存在）＝design §1 の「同 48/12 beam 族の模倣/ExpIt は beam-tie が構造天井」を **3-config で確証**。**∴ dN<0 は eval beam より厳密に強い信号源が必須**。A 未決①（CP-SAT 小群厳密教師）を最安エスカレーションとして昇格 → **但し §13 Step-0 で実規模 infeasible と確定**（honban dest 群は全て 25–64 item、CP-SAT は N=1 自明領域のみ OPTIMAL、N≥2 で n≥20 証明不能＝D-2 `8a0eb75` と同一の壁を教師用途で再確認）。**安価 2 路（B/ExpIt 天井 + A CP-SAT 実規模 infeasible）が両方閉**。B/ExpIt 路は dN<0 として exhausted（floor 改善＝B-clean 0.083 が成果）。**dN<0 残存ルートは重量級の C（真目的 RL）/ D（学習 improvement operator）のみ**（§13、新規 workstream・卒研トラック判断要）。

## 11. 実証ログ — B 本走（未決② 実装後 2-fold, 2026-05-18）

**設定**: `b_honban.py`。π0 = A §3 強教師 BC を collect_paths から bootstrap（env `TEACHER_BEAM_WIDTH=48 TEACHER_BRANCH=12 TEACHER_2SWAP_STEPS=2`）。hyper: iters≤6・alpha 0.75・replay 3・val-frac 0.3・base-ratio 1.0・reg-alpha 1e-3・seed 1234・patience 2。物差し=`beam_ref.json`（48/12 `beam_search_strong`, §2 i）, certified-LB（§2 iii）併記。決定的（seed 固定 + policy_guided_solve 決定的）。

| fold | hold-out | 選択モデル | base extract | mean_dN | per-inst dN | disq |
|---|---|---|---|---|---|---|
| A | suite[0:6] hard_01..06 | **iter0 = π0**（ExpIt 全反復 内部 val で π0 未超→early-stop） | 3761 rows / 475s | **0.833** | [0,2,0,1,1,1] | 0 |
| B | suite[10:16] hard_11,12+std4 | **iter3**（内部 val 11.333→11.000） | 3197 rows / 421s | **0.5** | [1,1,0,0,0,1] | 0 |

**監査済所見（fold毎 single-run、保守解釈・Fold-A-only headline 回避）**:

1. **dN<0 は 0/12 hold-out inst で未達**。2-fold avg 0.667。**2-fold ゲート FAIL（dN<0）**。
2. **A §3 robust floor（Fold A 0.0 / Fold B 0.333）を両 fold で下回る**。Fold A は hard_02 +2 / hard_04,05,06 +1（A §3 は全 tie だった）。
3. **主交絡 = 弱化 π0**: `--bootstrap-pi0` は val 切出後 7/10 inst で π0 を再訓練。A §3 Fold A model は全 10 inst 訓練で dN 0.0。∴ 本走は「ExpIt が A §3 floor を越えるか」でなく「bootstrap+ExpIt が *自身の弱化 π0* を越えるか」を測ってしまった（混同回避必須）。Fold A は弱化 π0 すら ExpIt が回復できず（選択=π0）, Fold B は ExpIt が内部 val を 0.333 inst 改善も hold-out へ非転写。
4. **未決②機構自体は正しく動作**: Ranked-Reward 分位ゲート（iter 毎 accept 7→2/7 等, データ駆動）・有界 replay（base/expit 行 balanced 3197/3197 維持）・**held-out 選択が両 fold で退行 iter を正しく不採用**（naive smoke 1.167 過適合は再現せず＝防御は機能）。**過適合防止 ≠ beam 超え**。
5. **certified-LB sanity**: Fold A hard_01..06 gap 2-3・Fold B hard_11/12 gap 3-4（非縮退）＝dN<0 の頭余地は証明上あり。但し Fold B std 3件 gap 0（beam 証明上最適, そこで dN<0 構造的に不可）＝Fold B mean_dN は構成上ほぼ 0 にピン留め（A §3 Fold B 0.333 が底だった理由を裏付け）。失敗は noise-impossibility でなく**手法の天井**（design §1: eval beam の模倣/合成は ceiling、policy-guided beam も同 48/12 beam 族ゆえ継承の可能性）。
6. **教訓（再挑戦防止）**: 小プール（10 inst）では「強い π0」と「クリーンな held-out val」が両立しない。bootstrap π0 を既定にすると A §3 比較が交絡。**B の真の検証は π0=A §3 フル固定**（§12）。spike→本走 規律で時間溶解前に設計要件（π0 固定 / val-frac 再考 / patience）を確定。

## 12. 実証ログ — B-clean（π0=A §3 フル固定で交絡除去, 2026-05-18）

**設定**: `b_honban.py --pi0-mode a3full`。fold ごとに `r1_train --model mlp`（強教師 env 48/12+2swap）で fold 全 train 10 inst から **A §3 フルモデル**を再生 → `coevo_expit --no-bootstrap-pi0 --model-in <それ> --val-frac 0.3`（内部 val は ExpIt 反復選択*専用*。val 3 inst は A §3 訓練に含むため選択信号に軽い楽観バイアスだが headline dN は **untouched hold-out** で honest ＝交絡除去）。hyper/物差しは §11 と同一。

| fold | A§3 regen | base | 選択 | mean_dN | per-inst dN |
|---|---|---|---|---|---|
| A | 793s | 3791/466s | iter6（val 10.0→9.333） | **0.0** | [0,0,0,0,0,0] |
| B | 631s | 3216/420s | iter2（val 10.667→10.333, early-stop p2） | **0.167** | [0,0,0,0,0,1] |

**全 run 比較（mean_dN, 物差し=48/12 beam_ref）**:

| 構成 | Fold A | Fold B | 2-fold avg | below-beam |
|---|---|---|---|---|
| A §3 純BC（フル） | 0.0 | 0.333 | ~0.167 | 0/N |
| B bootstrap（交絡 π0） | 0.833 | 0.5 | 0.667 | 0/12 |
| **B-clean（π0=A §3 フル）** | **0.0** | **0.167** | **0.083** | **0/12** |

**監査済所見（保守解釈）**:

1. **交絡を確証・除去**: bootstrap の悪化（0.833/0.5）は `--bootstrap-pi0`×小プールの π0 弱化が主因と確定（B-clean で Fold A 0.833→0.0・Fold B 0.5→0.167 と回復）。
2. **B-clean = 記録上最良の robust floor**（2-fold avg **0.083** < A §3 純BC ~0.167 < bootstrap 0.667）。**Fold B で ExpIt が hard_11/hard_12 を above-beam → beam-tie(0) まで押下げ 0.333→0.167**（残 +1 は case_small_many のみ、std 3 件は gap0 で正しく 0）、Fold A は 0.0 維持・退行ゼロ・disq0・greedy 速度。**∴ 交絡を除いた ExpIt は純 BC の floor を 2-fold で実改善する**（modest だが robust）。
3. **但し dN<0 は 0/12 で未達**: 全 hold-out inst で 48/12 beam を**厳密に下回らない**（Fold A 全 tie / Fold B 残 +1）。**2-fold ゲート FAIL（dN<0）**。
4. **3-config 確証 — beam-tie が構造天井**: A §3 純BC・B bootstrap・B-clean の **3 独立構成すべてで below-beam 0/N**（certified-LB で頭余地 gap 2-4 は存在＝noise-impossibility でない）。**模倣 + ExpIt を同 48/12 beam 族内で回す限り beam-tie が上限**（design §1 の thesis を 3-config で確証）。ExpIt の自己改善は **floor を beam-tie へ robust に押し上げる**が **ceiling（beam 超え）は破れない** — 信号源が eval beam を超えないため（§1）。
5. **決定的含意**: B/ExpIt 路は **dN<0 ルートとして exhausted**（floor 改善の成果＝B-clean 0.083 として確定・記録）。dN<0 には **eval beam より厳密に強い信号源**が必須:
   - **A 未決①（CP-SAT 小群厳密教師）**＝最安エスカレーション（証明付き最適 > beam を教師に。`cpsat_model` は count のみ→solver 解抽出+trace-replay 実装が前提）。
   - **C（真目的 RL）**＝beam を模倣せず箱数を直接最適化（POMO/AR2L/Preference-Opt）。
6. **教訓**: 純 BC も ExpIt も「eval beam の模倣/合成」である限り原理的に beam-tie 天井（§1）。**floor を上げる工学**（未決②・交絡除去）と **ceiling を破る研究**（より強い信号源）は別物。B-clean はその境界を 3-config で実証的に確定した。

## 13. 実証ログ — A 未決①（CP-SAT 厳密教師）Step-0 で実規模 infeasible（再挑戦防止記録, 2026-05-18）

B-clean 完了（§12）で dN<0 本命を A 未決①（CP-SAT 小群厳密教師＝証明付き最適 > beam を BC 教師化）へ昇格。**フル実装（solver 解抽出 + trace-replay）前に Step-0 で CP-SAT の実規模 tractability を実測**（[[feedback_ml_repo_trial_template]] / `8a0eb75` スパイク必須規律）→ **infeasible と確定、実装中止**。

**測定**:
- honban suite 全 16 inst の dest 群サイズ＝**全て 25–64 item**（最小 25、ヒストグラム: 25×2,26×2,…,64×1）。25 未満の dest 群は**存在しない**。
- `cpsat_model.solve_min_containers(with_support=True, time_limit=30s)` を実 item で n=6..25 掃引: **n≤16 は OPTIMAL だが全て N=1**（小 item が 1 コンテナに収まる自明解＝beam/greedy も同値, 教師価値ゼロ・dN 寄与 0）。**n=20,25 は FEASIBLE 止まり・certified=False（30s）**。＝CP-SAT が **OPTIMAL を証明できるのは N=1 の自明領域のみ**、N≥2（コンテナ最小化が意味を持つ唯一の領域）では n≥20 で証明不能。

**結論（dead-end・再挑戦防止）**: **非自明（N≥2）かつ CP-SAT-OPTIMAL-tractable な dest 群は実規模に 1 つも存在しない**（honban 最小 25 ≫ 自明 N=1 領域上限 ~16, かつ n≥20 で証明不能）。D-2 `8a0eb75`（CP-SAT 双対が dest 26–54 で自明値1へ縮退）と**同一の壁**を、今度は *教師* 用途で確認。∴ A 未決①（CP-SAT 厳密教師）は実規模で **infeasible**＝solver 解抽出 + trace-replay の実装は**着手しない**（適用先ゼロ）。sub-group へ heuristic 分割すると「厳密」優位が消え beam 同等以下（beam は群全体 multi-ordering keep-best 済）＝意味なし。synthetic small への curriculum/transfer も、証明可能な小 inst が全て自明 N=1＝**転写すべき非自明最適信号が源で存在しない**ため同様に dead。

**dN<0 の残存ルート**: 安価 2 路（B/ExpIt §12 ＝ beam-tie 天井 / A 未決① ＝ CP-SAT 実規模 infeasible）が**両方閉**。残るは重量級のみ:
- **C: 真目的 RL**（POMO/AR2L/Preference-Opt、beam も教師も模倣せず箱数を報酬で直接最適化＝tractable な厳密 oracle 不要）。Colab GPU、AR2L 真共進化＝卒研トラック相当。
- **D: 学習 improvement operator**（Neural LNS / 学習 destroy-repair）。構築天井と独立に解空間で beam 出力を改善＝ceiling と直交。中量・hedge。

＝**dN<0 は「安価エスカレーション」では到達不能と実証的に確定**（B/ExpIt 天井 + CP-SAT 実規模 infeasible の 2 点）。C/D は新規 workstream・明示的なプロジェクト判断（卒研トラック化）を要する。

## 14. 決定記録 — D-2（敵対生成器 regret 目的）= (b) beam 基準で受容（2026-05-19）

**未決③（§7）の確定**。ユーザー判断「授業トラックを締める」を受け、D-2（敵対生成器の regret 目的を beam 相対 → 証明付き下界ギャップへ差し替える当初案）を **(b) beam 基準＋限界明記で受容**に確定。(a) 強 MILP で下界 tighten は不採用。

### 決定

- 敵対生成器の regret 目的は **beam 相対を維持**: `ΔN = N_GA − N_beam_ref`（`beam_ref` = 48/12 `antagonist.beam_search_strong`, §2 i）＋辞書式 surrogate（`design.md §2`）。D-2 当初案（regret を perdest 証明付き下界とのギャップに置換）は **不採用**。
- certified LB（`rui/exact/per_destination_lb`, Martello–Toth L2）は **regret *目的* には使わず**、§2 iii の **sanity / 頭余地存在証明**として非縮退 small dest 群限定で併記運用（現状の §8 所見5・§10・§11 所見5 と同じ）。

### 前提崩壊の根拠（再掲・集約）

- `8a0eb75`（feat/rui-exact）: 部分支持緩和 CP-SAT 双対下界を実装・正当性は証明（syn_one/two OPTIMAL、トリップワイヤ `support_LB ≤ beam_N` 通過）したが、**実 honban dest 群（26–54, 全 25–64 item）で 90s FEASIBLE 止まり・双対下界が自明値 1 へ縮退 → `max(L2,dual)=L2` で tighten ゼロ**（現 `perdest_lb` と完全同一）。
- §13（A 未決① Step-0）: 同一の壁を *教師* 用途で再確認（CP-SAT OPTIMAL 証明は N=1 自明領域のみ、N≥2 で n≥20 証明不能）。
- ∴ D-2 のギャップ目的は実規模で **beam 相対とほぼ同値か非情報的**＝差し替える価値が構造的に存在しない。

### (a) を棄却する理由

強 MILP での下界 tighten は (i) 高労力、かつ (ii) `8a0eb75` が縮退は **構造的**（支持制約を正しく緩和しても双対 LP 根が dest 26–54 で弱く L2 へ縮退）と *実証*済 ＝ 期待利得が構造的にゼロ近傍。費用対効果で不可。スパイクで本実装前に確定済（再挑戦防止）。

### 受容に伴う既知の限界（明記が受容の条件）

1. **「hard」は eval beam 相対であって証明付き最適相対ではない**。生成 instance は "48/12 beam に対して GA が詰まる" 難所であり、絶対的最適からの乖離は実規模で未測。
2. **eval beam 自身が certified LB sanity で 2–4 箱 非最適**（非縮退 small dest 群で確認, §8 所見5 / §10）。∴ 絶対 hardness 天井は実規模で不明のまま（noise でなく計測手段の構造的限界）。
3. ただし**物差しの一貫性は保たれる**: cockpit/eval ランキングも beam 相対（§2 i）＝生成・評価・配布（`official_adv_hard`）が同一基準。certified LB は目的でなく頭余地存在証明として一貫運用。

### 波及

- research-track の dN<0 結論（§12 = 模倣/ExpIt は beam-tie 構造天井 / §13 = CP-SAT 実規模 infeasible）と完全整合。D-2 を beam 相対で確定 ＝ 敵対レーン（レーン2）の物差しが安定し、チーム配布の `official_adv_hard` ベンチも本基準で確定。
- ideamap `proj-vanning-layout.md` レーン2/レーン3 の D-2 タスクは本 §14 を SSOT として resolved（状態行のみ参照）。

## 関連

- ideamap `[[proj-vanning-layout]]` レーン3 / `[[know-adversarial-instance-3dbpp]]` レーン3実証ログ（SSOT）
- `[[know-prediction-error-as-preference-signal]]`（regret/予測誤差の横断同型 — shaping 設計の理論背骨）
- `REVIEW_adversarial_lane_2026-05-16.md` §6（強DAgger 監査値 0.5 の根拠）
