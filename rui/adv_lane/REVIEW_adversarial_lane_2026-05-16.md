# 敵対レーン (Lane 2 + 3) 方法論・結論 独立監査 REVIEW

監査: Claude (OpenCode 委譲 3 連敗→Claude 直接 fallback) / 2026-05-16
証拠一次資料: `_evidence_digest.md`（同ディレクトリ、file:line 付き factual digest）
監査対象主張: `scripts/opencode/briefs/vanning_adv_lane_review_brief.md` の被告陳述

---

## 0. 一段要約（歯に衣着せず）

**Lane-2（授業トラック＝敵対的貨物生成 A/B）は堅牢** — 全 headline を `trajectory.csv`
から独立再計算し proj-note と完全一致。ここは信用してよい。

**Lane-3（研究トラック＝decoder 学習化 R0→R2）は「方向性は示唆的だが証拠が脆い探索」**。
headline（offline BC で floor 0.833 が 2-fold 頑健 / cheap DAgger 棄却 / strong DAgger
で ceiling 到達）は次の5点で**結論として確定したと言える状態にない**:
(1) per-fold N=6・1 instance=0.167dN で、主張差は 2〜3 instance 差に過ぎず、
(2) ga_bench が numpy seed 未固定で GA は確率的（progress.log で 10.438⇄10.562 揺れ実測）
＝1-instance 差は RNG ノイズ帯と分離不能、(3) Lane-3 のほぼ全 headline に**生存する
per-instance artifact がゼロ**（progress.log の集約1行と commit message のみ）、
(4) logreg→MLP「線形容量が原因」は**コード上交絡**（balanced 有無 / MLP seed 未固定）、
(5) dN の物差し（strong beam）は teacher trace・R2 oracle と**同一ヒューリスティック族**
＝非独立参照。加えて (6) **監査対象の brief 結論が HEAD で陳腐化**（後述 矛盾2）。

**最も信用できない結論**: 「R1-MLP hardening = 2-fold で headline 確定、単一 split の
運でない」。6 件窓 ×2 fold・dN 粒度 1-instance・GA RNG 未固定では「運でない」は原理的に
立証不能。これは *誤り* という意味ではなく、*主張の確信度が証拠を大きく超過* している。

---

## 1. 再構成マップ（コミット/コードで確認した実体）

`feat/rui-coevo` は `feat/rui-adv-lane` を厳密に superset（共通 tip `e00ec13`）。
`auto/rui-ga-overnight` は `e00ec13` から分岐し GA/beam 参照解を足す（dN 基準線の出所）。

```
Lane2  9cfb288 scaffold(regret/antagonist/generator/loop) → ed5146f A:regret整形
       → 39e14f4/722a0e5 B:31-cat → 396433c bugfix → e00ec13 (Lane2 tip)
Lane3  5d56ad7 R0 pack_items_beam → bfb7804 R1 logreg → bb1365f R1 cross-val(MLP Fold A/B)
       → 9ae85d9 R2 DAgger-lite(floor↓ 0.833→1.167) → c1f8031 R2 strong(ceiling 0.833→0.0 と主張)
```

eval suite = honban hard 12 + std 4 = **真 N=16**（ga_bench.py:77,80-85,91、baseline.json
`"instances":16`）。Fold A=suite[0:6]=hard_01..06 / Fold B=suite[10:16]=hard_11,12+std4、
**instance 単位 disjoint＝リークなし**（r1_train.py:52-54, r1_teacher.py:256-281、コード明白）。
各 fold eval **N=6、0.1667 dN = 厳密に 1 instance**。

## 2. 主張別 Verdict

凡例: ✅supported / ⚠️overclaimed（観測は真だが主張が証拠を超過）/ ❌unsupported /
🔍needs-more-evidence（実証決着待ち）

| # | 主張 | Verdict | 根拠 (file:line) / 反証・懸念 |
|---|---|---|---|
| L2-scaffold | regret 辞書式+PAIRED gate、pytest 8/8 | ✅機構 / 🔍件数 | gate は regret.py:34-41 で仕様通り実装、ε lex-safety test_regret.py:55-63。但し pytest 出力 artifact 無し、"8/8" は未検証（`.pyc` 収集痕のみ） |
| L2-A | 整形で dN≥1 1→10、mean 0.036→0.185 | ✅supported | `runs/step1_{cma,shaped}` trajectory.csv から独立再計算: 0.0357→0.1853、1→10。loop が純 r を保存・shaped は CMA のみ (loop.py:341-343) も確認 |
| L2-B | 31-cat dN_max1→3、17%→63%、→0.990、none0 | ✅supported（"17%"は表示丸め） | step2_cat31 再計算 mean 0.9902/dN_max3/r≥1 62.5%/none0。"17%" は step1_shaped 15.6% の丸め（矛盾3） |
| L2-honban | 180eval none0、entropy平坦、dN_max3、dN≥2 26%、0.967 | ✅supported | trajectory 181行=180eval、mean 0.9672、dN_max3、dN≥2=46/180=25.6%、gen_summary entropy 0.692-0.718。自己整合 |
| L2-bugfix | id採番/size_type=dN不変 cosmetic | ✅supported | dN は `len(containers)` 基準で id 不使用 (ga_bench.py:166)、size_type は geometry に不介入 (design_catalog31.md:136)。コード明白に成立 |
| L3-R0 | beam decode 6inst 1.33→1.00、4/6改善、hard_01悪化、~6×遅 | ⚠️/🔍 | progress.log は **beam が集約 REGRESS**（10.833 vs greedy 10.438）。per-inst 1.33→1.00・hard_01悪化は**保存 artifact ゼロ**（last_bench 上書き）。遅さは **~4×（496/122s）で ~6× は誤**（矛盾1） |
| L3-R1 logreg→MLP | logreg dN2.33→MLP 0.833、原因=線形容量、模倣精度≠デコード | ⚠️overclaimed | 退行は事実 (progress.log 12:59/13:20)。但し **交絡**: logreg=`class_weight=balanced`/MLP=無 (r1_train.py:72 vs 75)、**MLP に random_state 無**＝seed 非決定、early-stop 無。"線形容量が原因" は分離されていない。val top1 は stdout のみで未保存＝"模倣精度≠" 検証不能 |
| L3-R1-MLP hardening | 2-fold、Fold A 1.33→0.833 / B 0.667→0.333、悪化ゼロ、運でない | ⚠️overclaimed / 🔍 | リーク無は ✅（コード明白）。但し **N=6/fold・1 inst=0.167dN**＝主張差は 3/6・2/6 instance。`scorer.joblib` 単一上書きで per-fold model・per-inst 悪化ゼロ・改善 instance 同一性 **artifact ゼロ**。GA RNG 未固定で 1-inst 差は noise 帯。「運でない」は 2 窓 6 件では原理的に未立証 |
| L3-R2 DAgger-lite | 安価 oracle DAgger が floor↓(0.833→1.167)→cheap DAgger 棄却 | ❌causal unsupported | 退行 1 回は事実 (progress.log 14:06)。但し **4 交絡未分離**（弱oracle / sparse 1-order labeling max_label=30 / fresh-retrain seed noise / horizon-width）、単一走・反復なし。"弱 oracle が原因" の帰属は立証されていない |
| L3-R2 strong | strong DAgger で Fold A 0.833→0.0(6/6 beam一致)、B 0.333→0.167 | 🔍 commit-message-only | **Fold B 0.167 のみ artifact 生存**(last_bench 14:47:48)。**Fold A 0.0 / 6/6 一致は progress.log 1行 + commit message のみ**、単一 seed・random_state 無。**最強の主張が最弱の証拠保全**。これが HEAD の到達点（矛盾2） |
| Repro | GA (gen,idx) seed 並列再現、結果不変、mode collapse 無 | ⚠️partial | 並列 `_eval_one` は決定的 reseed (loop.py:196-199) ✅。但し **直列パスは reseed せず**、ga_bench は numpy seed 未固定 (ga_bench.py:142-143)＝GA 確率的。"結果不変" の serial-vs-parallel artifact 無。mode collapse 無は entropy 平坦で ✅ |

## 3. 方法論の穴 Top 5 ＋ それを塞ぐ最小実験（コスト見積、未実行）

1. **dN の物差しが非独立**（最重要・横断）: dN = GA vs **strong-beam ヒューリスティック**
   (width48/branch12/4 ordering、antagonist.py:162-166)＝最適下界でない。しかも
   **teacher trace (r1_teacher.py:187)・R2 lite oracle (r1_dagger.py:130)・dN 物差しが
   全て同一 `_partial_lex_key` 族**。学習デコーダを「同じヒューリスティックへの一致」で
   訓練し「同じヒューリスティックとの差」で評価＝真のコンテナ最小化でなく *beam 模倣度*
   を測っている恐れ。→ 独立参照（小規模 MIP / 別系統 solver）で dN を 6inst 再採点（~stretch）。
2. **N=6/fold・GA RNG 未固定で signal と noise 分離不能**: → ga_bench に numpy seed 固定を
   足し、Fold A/B を seed 固定で再走、per-instance dN を greedy/learned で dump、符号検定
   （~30min）。これが headline の生死を決める核。
3. **Lane-3 headline の per-instance artifact が消滅**: 全主張が progress.log 集約1行頼み。
   → 上記再走時に per-row JSON を必ず永続化（運用改善、追加コストゼロ）。
4. **logreg→MLP 因果が交絡**: → MLP+balanced / logreg−balanced / 両者 random_state 固定で
   val-top1+dN を比較（~40min）。"線形容量" 説の真偽を分離。
5. **HEAD の最強主張 (strong DAgger Fold A=0.0) が transient stdout のみ**: → strong DAgger
   Fold A 再走、per-inst ga_containers vs beam_N を dump（~20min）。

## 4. 総合判定

- **Lane-2 は授業納品として信用に足る**。A（頻度）+B（深さ）相補は独立再計算で実証済、
  bugfix の cosmetic 主張もコード明白に成立。ここに追加実証は不要。
- **Lane-3 は「探索の地図」としては価値があるが、結論として引用してはならない**段階。
  特に提案書・卒研・対外資料で「offline BC で floor が立つ/DAgger は安価近道不可」を
  *確定事実* として書くのは時期尚早。N・RNG・artifact 保全・参照独立性・HEAD 整合の
  5 点が未解決。**方向性の仮説**としてなら妥当。
- **監査の前提自体が陳腐化**: brief（=proj-note 由来）の R2 結論「cheap 棄却・strong/RL
  のみ」は HEAD commit `c1f8031`（strong DAgger が既に ceiling 到達と主張）に追い越されて
  いる。proj-vanning-layout.md の Lane-3 完了履歴も同様に要更新。

## 6. 実証決着 — bounded ablation (2026-05-16 17:41-18:05, `runs/audit_2026-05-16/`)

争点を実走（各 fold MLP 再訓練→learned を **2 回**＋greedy、logreg Fold A、strong
DAgger Fold A）。per-instance dN を全保全（穴#3 解消）。

| 主張 | 監査前 | 実証決着 | 根拠 (per-inst dN) |
|---|---|---|---|
| 学習デコーダ > greedy 両 fold | ⚠️/🔍 | ✅ **支持** | A: greedy1.333→learned0.833 / B: 0.667→0.5 |
| 全 inst 悪化ゼロ | 🔍 artifact 無 | ✅ **支持** | A greedy[2,2,1,0,2,1]→learned[2,2,0,0,0,1] / B [1,2,0,0,0,1]→[1,1,0,0,0,1]、悪化 instance ゼロ |
| 再現性（GA ノイズでない） | ⚠️ RNG 未固定 | ✅ **支持** | **run1==run2 が両 fold で完全一致**（A=[2,2,0,0,0,1]×2, B=[1,1,0,0,0,1]×2）。N=6 で GA ノイズは顕在化せず |
| Fold A floor = 0.833 | 🔍 | ✅ **正確に再現** | 0.833, [2,2,0,0,0,1] |
| Fold B floor = 0.333 | 🔍 | ❌ **再現せず＝0.5** | 改善幅は主張の約半分（greedy0.667→0.5、主張は→0.333） |
| logreg2.33→MLP0.83 gap | ⚠️ | ✅ 現象再現／一部交絡解消 | logreg 2.333[3,3,2,1,3,2] vs MLP 0.833 を正確再現。**MLP seed 非決定の懸念は実証的に棄却**（run1==run2＝決定的）。`class_weight=balanced` 非対称の交絡は**未解消**（balanced 統制版は未実行） |
| 「線形容量が原因」 | ⚠️overclaim | 🔍 据置 | 現象は頑健だが因果分離（balanced 統制）は未実行。"原因=容量" は依然未立証 |
| **strong DAgger Fold A → 0.0 (HEAD c1f8031 最強主張)** | 🔍 commit のみ | ❌ **再現せず＝0.5** | [0,1,0,1,0,1]。0.833→0.5 と改善はするが **ceiling(0.0) 到達は誇張**。「classic DAgger が理論通り機能」は過大 |

**決着の要旨**:
- **方向性は監査前評価より強い**: 「offline-BC 学習デコーダが greedy を再現性よく・悪化ゼロで上回る」は **実証支持**（2 fold とも同方向、run1==run2、per-inst 検証済）。Lane-3 の中核仮説は生きている。
- **個別数値は楽観方向に誇張**: Fold B 0.333 は実際 0.5、HEAD 最強の strong-DAgger「0.0 ceiling 到達」は実際 0.5。**方向は真・到達点の主張は誇大**。「ceiling 到達」ナラティブは偽。
- **N=6×2fold の検出力限界は残存**: 再現性は確定したが、汎化（この 12+4 instance を超えて）は未検証。
- 監査前の「最も信用できない結論（2-fold で運でない）」→ **部分的に好転**: 運（GA ノイズ）は実証的に否定。但し「2 窓 6 件を超えた一般性」は依然未立証で、これと strong-DAgger 誇張が新たな最重要注意点。

## 5. 付録: 読めなかった一次資料

pytest pass/fail（rule で未実行、`.pyc` 収集痕のみ）/ R0・R1-logreg・R1-MLP-foldA・
R2-lite・R2-strong-foldA の per-instance dN（`last_bench.json` 上書き、Fold-B strong のみ
生存）/ logreg vs MLP の val-top1（stdout のみ）/ serial-vs-parallel "結果不変" 比較（artifact 無）。
詳細根拠は `_evidence_digest.md` §2-§5。
