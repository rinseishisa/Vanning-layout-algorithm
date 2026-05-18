# Handoff — 未 push feat ブランチ 3 本（2026-05-19 整備）

> 目的: B→A→C+D1 キャンペーン（2026-05-18）と spike（5-19）の成果が **2 repo に分散した 3 本の未 push feat ブランチ**として滞留している。チーム調整して push / PR / merge / GCP 反映するための SSOT。
> push 自体は **チーム調整事項のため未実行**（本書は判断材料）。SSOT: ideamap `proj-vanning-layout.md` レーン2/3 + `rui/adv_lane/design_coevo.md`。

## 0. 全体像（2 repo / 3 branch）

| # | repo | branch | tip | base | commits | 規模 | merge リスク |
|---|---|---|---|---|---|---|---|
| 1 | `vanning-algo` | `feat/rui-exact` | `8a0eb75` | `feat/rui-ga@33931d1` | 2 | +1137 / 11 files（全 `rui/exact/` 新規） | **低**（greenfield、共有ファイル無改変） |
| 2 | `vanning-algo` | `feat/rui-ga-portfolio` | `4df9500` | `feat/rui-ga@33931d1` | 1 | +40/-1 / 2 files | **低**（加算的） |
| 3 | `vanning-eval` | `feat/adv-bench-tab` | `9536589` | `main@d57468e` | 6 | +36815/-66 / 38 files | **中**（canonical 基盤 5commit 内包 + main が origin から behind 9） |

ブランチトポロジ（vanning-algo）:

```
main@14c5642 ──(PR#27 base)
   └ origin/feat/rui-ga@33931d1  ← 授業主納品幹（push 済・PR#27）
        ├ feat/rui-exact@8a0eb75        (#1)
        └ feat/rui-ga-portfolio@4df9500 (#2)
   └ feat/rui-adv-lane@e00ec13 ← origin 済・**PR #31 オープン**（base main、別系）
```

- #1/#2 は **授業幹 `feat/rui-ga@33931d1` の上**に乗る。`feat/rui-ga` は origin push 済（PR #27 base main）。よって #1/#2 の PR base は `feat/rui-ga`（差分最小）か、`feat/rui-ga` merge 後に `main` か をチームで決める。
- `feat/rui-adv-lane`（PR #31, レーン2 敵対生成 5commit）は別系で既に origin・スコープ重複なし共存。

---

## 1. `feat/rui-exact` — beam vs 証明付き最適（certified LB + CP-SAT）

**repo**: `vanning-algo` / **base**: `feat/rui-ga@33931d1` / **tip**: `8a0eb75`

```
8a0eb75 feat(rui/exact): correct partial-support relaxation + spike finding (NOT viable at scale)
b5dec35 feat(rui/exact): Task B — beam vs optimal (certified LB + small-instance CP-SAT)
```

**変更**: `rui/exact/` 新規モジュールのみ（`__init__/lower_bounds/cpsat_model/slice_instances/bench_table/verify` + `tests/` + `runs/*.json`）。**既存共有ファイルへの改変ゼロ**＝コンフリクトリスクほぼ無。

**何を / なぜ**:
- Martello–Toth L2 **証明付き下界** + 小規模 CP-SAT 厳密解で「敵対ベンチ hard が本当に難しいか」を客観検証する物差しを新設。
- 結果: honban hard 12 件で `beam_N 9–10` vs 証明下界 `6–8` ＝ **beam は証明された最適から最大 2–4 箱**。126item/3dest 規模は厳密解不可ゆえ下界、合成は CP-SAT で OPTIMAL 証明。
- `8a0eb75` は **意図的に保存した negative result**（部分支持緩和 CP-SAT 双対下界は正当だが実規模 dest 26–54 で自明値へ縮退＝tighten 不可）。再挑戦防止記録。production LB/bench_table へは**非統合**（L2-only のまま）＝この commit はドキュメント価値で残す。

**検証済**: `rui/exact/tests/` pytest（`test_lower_bounds` / `test_cpsat_small`）、syn_one/two CP-SAT OPTIMAL、トリップワイヤ `support_LB ≤ beam_N` 通過。

**PR ドラフト**:
- title: `feat(rui/exact): beam vs 証明付き最適 — certified LB + small-instance CP-SAT`
- base: `feat/rui-ga`（推奨。`feat/rui-ga` merge 後なら `main`）
- body 要点: 上記「何を/なぜ/検証」。`8a0eb75` は dead-end の保存記録であり production 非統合と明記。`rui/exact/` 自己完結ゆえ単独 merge 可。

---

## 2. `feat/rui-ga-portfolio` — R1-MLP decoder の portfolio 昇格（Task C）

**repo**: `vanning-algo` / **base**: `feat/rui-ga@33931d1` / **tip**: `4df9500`

```
4df9500 feat(rui/adv_lane): Task C — promote R1-MLP decoder via pack_items_portfolio
```

**変更**: `rui/adv_lane/coevo_decoder.py` (+34) / `rui/adv_lane/ga_bench.py` (+7/-1)。加算的・小。

**何を / なぜ**:
- `pack_items_portfolio` ＝ greedy と learned デコードの **`fitness_key` が良い方**を採用。learned 不在/失敗時は greedy へ縮退＝**構成上 greedy 以下にならず失格も増えない（REGRESSED が構造的に不能）**。
- 実 `vanning_eval` で **IMPROVED**: `mean_containers 10.625→9.75` / `dN(対beam) 1.188→0.312` / `disq 0→0` / 421s（天井 1500s）。

**ランタイム依存**: learned 経路は R1-MLP scorer 成果物（`scorer.joblib` 等）の存在が前提。**不在時は自動で greedy 縮退**するため本番投入の安全性は保たれる（PR body に明記推奨）。

**検証済**: 上記実 `vanning_eval` 数値（再現確認済、proj 完了履歴 2026-05-18）。

**PR ドラフト**:
- title: `feat(rui/adv_lane): GA decoder を portfolio 化（greedy∪learned）— 対 beam dN 1.188→0.312`
- base: `feat/rui-ga`
- body 要点: portfolio の安全性（greedy 縮退で REGRESSED 構造的不能）、IMPROVED 数値、scorer 依存と縮退挙動。

---

## 3. `feat/adv-bench-tab` — 敵対ベンチ配布 + cockpit タブ（Task A）

**repo**: `vanning-eval` / **base**: `main@d57468e` / **tip**: `9536589`

```
9536589 feat(adv-bench): Task A — distribute 12 hard instances + 敵対ベンチ cockpit tab
dac2070 chore(scoreboard): canonical 再検証で history を是正（非破壊 quarantine）
ace871d chore(scoreboard): 公式 canonical input 配置 + DESIGN/README 整備
5394856 feat(scoreboard): 汚染履歴の canonical 再検証マイグレーション
b02a887 feat(scoreboard): canonical 整合ゲートをランキングへ配線
c8d8c65 feat(canonical): content-hash + canonical registry + 整合ゲート
```

**変更**: 38 files, +36815/-66。内訳: `official_adv_hard_01..12/items_input.json`（12 hard、canonical・content-hash で source と一致＝改竄防止前提）/ `distribute/adv_hard_bundle/` チーム配布 bundle + README（GCP 投入フロー）/ `scoreboard/adv_bench_bounds.json`（Task B certified LB を canonical id へ re-key）/ `streamlit_app.py` 敵対ベンチタブ（既存 `_apply_canonical_gate`/`_rank_entries`/`_render_leaderboard_table` 再利用、下界差列、bounds 無しでも graceful）/ `tests/test_adv_bench_tab.py` 4 ケース。巨大 diff の大半は hard instance JSON。

**⚠️ merge 注意（チーム必読）**:
1. **canonical 基盤 5commit を内包**（`c8d8c65..dac2070`）。これは敵対タブの前提だが、他の vanning-eval 作業（`fix/canonical-gate@dac2070`、`feat/lexicographic-ranking`[origin: ahead 1]）と**重複の可能性**。merge 前に「canonical 基盤は誰の commit を正とするか」をチームで確定。
2. **`vanning-eval` の `main` が `origin/main` から behind 9**。本ブランチは古い `main@d57468e` から分岐。push 前に `origin/main` を取り込んで rebase/merge し、canonical 基盤の二重適用が無いか確認。
3. `scoreboard/history.json` +642（汚染履歴 quarantine マイグレーション）＝非破壊だが履歴ファイルを書き換えるので、運用中 scoreboard との整合をチームで確認。
4. untracked `vanning_eval_rui/docs/web_server_overview.html` は本 handoff 対象外（無関係・別途判断）。

**検証済**: commit の VERDICT PASS（schema load + hash identity + bounds shape + pytest、既存 `test_streamlit_score` 回帰込み）。

**GCP 反映手順**（人間 merge 後）:
1. チームが `feat/adv-bench-tab` を `origin/main`（または運用ブランチ）へ merge
2. GCP インスタンスで該当 repo を `git pull`
3. Streamlit/Caddy サービス再起動（`proj-vanning-eval-gcp` の systemd 化が未なら現行手順、systemd 化後は `systemctl restart <unit>`）
4. cockpit で「敵対ベンチ」タブ表示・下界差列・canonical gate 動作を目視確認

**PR ドラフト**:
- title: `feat(adv-bench): 敵対ベンチ 12 hard 配布 + cockpit タブ + certified LB 下界差`
- base: `origin/main`（rebase 後）。canonical 基盤の扱いをチーム確定後
- body 要点: 上記「変更/検証」+ ⚠️merge 注意 4 点 + GCP 反映手順。canonical 基盤 5commit が他ブランチと重複しないかの確認を merge 前提条件として明記。

---

## 4. 推奨 merge 順 / アクション（チーム判断材料）

1. **vanning-algo**: `feat/rui-ga`（PR #27）を先に main へ → その後 `feat/rui-exact` / `feat/rui-ga-portfolio` を base=main で PR（または `feat/rui-ga` base で先行レビュー）。両者は独立・低リスクゆえ順不同。
2. **vanning-eval**: `feat/adv-bench-tab` は **canonical 基盤の所有権確定 → origin/main rebase → 再検証**を経てから PR。最も調整コストが高いので先に着手判断を。
3. GCP 反映は #3 merge 後（手順は §3）。
4. push 自体は本書を提示してチーム合意後に琉生が実行。

> 補足: #1/#2 は `rui/` 配下自己完結で授業納品本体のリスクを上げない。#3 のみ vanning-eval 共通基盤に触れるため、レビュー資源は #3 に集中させるのが効率的。
