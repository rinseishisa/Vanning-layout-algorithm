# Evidence Digest — Vanning Adversarial Lane (Lane 2 + 3) Methodological Audit

Reconstructed 2026-05-16 from working-tree (`feat/rui-coevo` checked out) + git
`show`/`log` of `feat/rui-adv-lane` / `auto/rui-ga-overnight`. Factual digest only;
no verdicts. All citations are `file:line` against the working tree unless noted.

---

## 1. Reconstructed Work-Chain Map (timeline + deps, from commits/code)

Single linear commit chain on `feat/rui-coevo` (it strictly supersets
`feat/rui-adv-lane`; both share the same SHAs up to `e00ec13`). `auto/rui-ga-overnight`
branches at `e00ec13` and adds 3 GA/beam commits.

```
5504521 design.md (Lane2 scaffold spec)
9cfb288 scaffold: regret.py + antagonist.py + generator.py + loop.py + tests   ← Lane2 "pytest 8/8"
ed5146f A: regret shaping (shaped_fitness = r + lam·pressure)                  ← runs/step1_cma vs step1_shaped
39e14f4 B: theta31.py + catalog31.py (31-cat reparam, design_catalog31.md)
722a0e5 B: generator31.py (31-cat sampler) + size_type enum fix
396433c fix: antagonist container_id renumber + multi-order seeds             ← test_antagonist.py
e00ec13 loop: catalog switch + shaping + process-parallel (_eval_one)        ┐ shared tip Lane2
   ├── auto/rui-ga-overnight: 59a0d6e overnight GA + beam_reference_scale     │
   │    bb921dc abs-time ceiling   c1448bf GA beam-seed+memetic (-16% gap)    │  (dN baseline origin)
   │    f9ae935 要件定義書 §3.3                                                │
   └── feat/rui-coevo (research track = decoder learning):
        5d56ad7 R0: pack_items_beam + ga_bench --decoder switch
        bfb7804 R1: r1_teacher (beam-trace BC) + r1_train logreg + learned decode  ← logreg dN2.33
        bb1365f R1: cross-val (ga_bench --offset / r1_train --ho-start/--ho-len)  ← MLP Fold A/B
        9ae85d9 R2: DAgger-lite + cheap oracle → floor pushed down (0.833→1.167)
        c1f8031 R2: strong-oracle DAgger → claims ceiling reached (0.833→0.0 / 0.333→0.167)
```

Eval suite (`ga_bench.build_suite`, ga_bench.py:88-97): `hard_instances/honban_cat31/hard_01..12_g15_p12.json`
(N_HARD=12, ga_bench.py:77) **+ 4 std datasets** `case_{balanced,weight_bound,volume_bound,small_many}_seed42.json`
(ga_bench.py:80-85; all 4 confirmed to exist in `rui/datasets/`). **True eval N = 16.**

Fold split is **window-based** on this deterministic suite:
- Fold A = `suite[0:6]` = hard_01..06 (hold-out); train = `suite[6:16]` (10 inst). r1_train.py:52-54.
- Fold B = `suite[10:16]` = hard_11, hard_12 + the 4 std cases (hold-out); train = `suite[0:10]`. Cache `runs/r1/base_ho10_16.npz` confirms `_base_cache_path(10,16)` (r1_dagger.py:57-58).
- `last_bench.json` (ts 14:47:48) = decoder=learned, **instances=6**, rows = hard_11,12 + 4 std, mean_dN=0.167 → this is the Fold-B strong-DAgger result.

---

## 2. Per-Claim Evidence Table

| # | Claim (brief, UNVERIFIED) | What code/artifacts actually show (file:line) | Statically settleable? / experiment to settle |
|---|---|---|---|
| L2-scaffold | regret lex surrogate `dN+ε·dDev` + PAIRED gate; pytest 8/8 | `regret.compute_regret` exactly implements gate (a.dq→None, p.dq→dq_bonus, else dN+eps·dDev) regret.py:34-41; ε=1e-4 lex-safety asserted test_regret.py:55-63. Test files exist (test_regret 9 cases, test_antagonist 2, test_theta31, test_generator31). **"8/8" count not verifiable from artifacts** — no pytest output file; tests/__pycache__ shows tests *were* collected (`.cpython-311-pytest-9.0.3.pyc`). test_generator31 docstring line 4-5 says "not executed in this headless env". | Partially. Pass/fail count needs `pytest` run (≈30s, blocked by brief rule 4). |
| L2-A | shaping flips/lifts: dN≥1 1→10, mean r 0.036→0.185 | **CONFIRMED from trajectory.csv**: step1_cma mean_r=0.0357, r≥1 count=1; step1_shaped mean_r=0.1853, r≥1 count=10 (recomputed from `runs/step1_cma_g8p8/trajectory.csv` & `runs/step1_shaped_g8p8/trajectory.csv`, 64 evals each = 8gen×8pop). shaped_fitness regret.py:60-72; loop stores **pure r** in trajectory, shaped only fed to CMA (loop.py:341-343). | Settleable (done). Numbers match the proj note exactly. dN_max=1 for both (3-cat ceiling). |
| L2-B | 31-cat: dN_max 1→3, dN≥1 17%→63%, mean 0.185→0.990, none 0% | **CONFIRMED**: step2_cat31 mean_r=0.9902, dN_max=3, r≥1 rate=62.5%, none=0 (recomputed, 64 evals). theta31 normalized-entropy guard check_feasibility31 theta31.py:218-222 (H/ln31 ≥0.30). | Settleable (done). "17%" baseline = step1_shaped 15.6% (close). |
| L2-honban | honban_cat31_g15p12: 180eval, none 0% all 15 gen, entropy 0.69-0.72, dN_max=3, dN≥2=26%, mean r 0.967 | **CONFIRMED**: trajectory.csv = 181 lines = **180 evals** (15gen×12pop); none=0; mean_r=0.9672; dN_max=3; dN≥2 = 46/180 = **25.6%**; gen_summary mean_entropy 0.692-0.718 (loop writes per-gen). "curated hard 20" = `hard_instances/honban_cat31/` has 20 files (TOP_K_SAVE=20, loop.py:49,479). | Settleable (done). Self-consistent. |
| L2-bugfix | container_id renumber + size_type enum: "dN/regret unchanged cosmetic" | antagonist.py:200-209 renumbers `c.container_id=i` AFTER extend (dN uses `len(containers)` ga_bench.py:166, never the id) → arithmetically cosmetic. size_type: generator31._size_class buckets by volume (generator31.py:131, theta31 anchors), size_type unused in placement/scoring per design_catalog31.md:136. test_antagonist.py asserts id uniqueness. | Settleable (code-evident). Claim holds: dN counts containers by `len`, not id; size_type never enters geometry. |
| L3-R0 | beam decode: 6inst mean_dN 1.33→1.00, 4/6 improve, hard_01 worse, ~6× slow | Code path = `pack_items_beam` (coevo_decoder.py:37-57) reuses `_beam_search_for_group`. DEFAULT_BEAM_WIDTH=4/BRANCH=2 (coevo_decoder.py:33-35). progress.log 12:11 greedy IMPROVED 10.438; 12:41 **beam REGRESSED mean_containers 10.833 wall 496s** (vs greedy 122s ≈ 4×, not 6×). **Per-instance 1.33→1.00 / hard_01-worse NOT in any saved artifact** (last_bench.json overwritten; only aggregate progress.log lines survive). | NOT fully. Per-inst dN breakdown & "hard_01 worse" need a re-run `ga_bench --decoder beam --limit 6` (≈8min, 1 bench). |
| L3-R1-logreg | logreg dN 2.33 (worse); same pipeline MLP(32,16) dN 0.833; "cause=linear capacity, val top1 unchanged but dN swings" | progress.log: 12:59 learned **REGRESSED dN 2.333** (logreg); 13:20 learned **REGRESSED dN 0.833** (MLP, mean_containers 10.667). r1_train.py:70-75 logreg=`LogisticRegression(max_iter=2000,class_weight="balanced")` vs mlp=`MLPClassifier((32,16),max_iter=400)`. **Both share StandardScaler Pipeline** (r1_train.py:84) → scaling NOT a confound. **BUT confounds remain**: logreg has `class_weight="balanced"`, MLP does NOT (r1_train.py:72 vs 75); MLP `max_iter=400` no early-stop/n_iter_no_change set (sklearn default early_stopping=False); **no random_state on MLP** (r1_train.py:75) → MLP fit is seed-nondeterministic. val top1 numbers only in stdout, not saved. | NOT settleable for the "linear capacity" causal claim. Needs ablation: MLP with class_weight=balanced; logreg without; fixed random_state; report val-top1 + dN each. ≈4 trains+4 benches (~40min). |
| L3-R1-MLP-hardening | 2 fold, Fold A 1.33→0.833, Fold B 0.667→0.333, zero per-inst regression, "not luck of single split" | progress.log: 13:20 learned dN 0.833 (Fold A, mean_containers 10.667 over 6); 13:34 learned **IMPROVED dN 0.333** (Fold B, mean_containers 9.167 over 6); 13:35 greedy dN 0.667 (Fold B baseline). Window split is **instance-level disjoint** (r1_train.py:52-54 `train=suite[:s]+suite[e:]`, hold-out=`suite[s:e]`; build_dataset extracts per whole-instance, r1_teacher.py:256-281) → **no instance leakage**. `runs/r1/scorer.joblib` is a **single file overwritten each train** — no per-fold model snapshot survives; "zero per-inst regression" & "which instances improved (same vs different across folds)" are NOT in artifacts. mean dN denominators are **6 instances** (instances=6 in last_bench / progress lines). | Partial. Leakage: settleable (none, code-evident). Statistical-power / "not luck" / per-inst regression / fold-overlap: NOT settleable — last_bench overwritten. Needs: re-run both folds, dump per-row dN for greedy & learned, McNemar/sign over 6 inst. ~4 benches (~30min). N=6 per fold = each 0.166 dN step ≈ 1 instance. |
| L3-R2-DAgger-lite | cheap oracle (trunc-horizon + `_partial_lex_key`) DAgger pushes floor down (Fold A 0.833→1.167); conclude reject cheap DAgger | progress.log 14:06 learned **REGRESSED dN 1.167** (mean_containers 11.0 / 6) = post-DAgger-lite Fold A. Oracle: `_oracle_label` strong=False path uses `seq=[item]+remaining[:horizon-1]`, `sel=_partial_lex_key`, width=DEF_WIDTH=12, H=DEF_HORIZON=8 (r1_dagger.py:106-133, 52-54). Aggregation = base BC (npz cache) + DAgger rows vstack, **retrain fresh MLP** `(32,16) max_iter=400` no random_state (r1_dagger.py:271-288). **Confound NOT isolated in code**: degradation could be (i) weak oracle, (ii) sparse labeling `max_label=30` 1 order only (r1_dagger.py:54,150-153), (iii) fresh-retrain seed noise (no random_state), (iv) horizon/width hyperparams. Only one lite run in progress.log; no repeat. | NOT settleable. The causal attribution "weak oracle" vs pipeline/hyperparam/seed is a single-run, multi-confound result. Needs: (a) repeat lite ×3 seeds, (b) hold oracle fixed vary aggregation, (c) fixed random_state. ~6 DAgger runs + benches (~1h). |
| L3-R2-strong | strong oracle DAgger reaches ceiling, Fold A 0.833→0.0 (6/6 beam-match), Fold B 0.333→0.167, "classic DAgger works as theory" | Commit c1f8031 message asserts this. `--strong` path: `seq=[item]+all remaining`, `sel=_final_lex_key` (r1_dagger.py:96-106). Only **Fold B** survives in artifact: last_bench.json 14:47:48 decoder=learned instances=6 mean_dN=**0.167** (hard_11,12+4std all dN=0 except case_small_many dN=1) → matches "Fold B 0.167". **Fold A "0.0 / 6/6 match" NOT in any artifact** (last_bench overwritten by Fold B run at 14:47; progress.log 14:37 learned IMPROVED dN 0.0 mean_containers 9.833/6 = the Fold-A strong-DAgger line, but the 6/6 beam-ref agreement detail is not stored). This is the **strongest claim** and rests on transient stdout. | NOT settleable for Fold A. progress.log 14:37 corroborates dN=0.0 aggregate (1 line) but not "6/6 beam_ref identical". Needs: re-run strong DAgger Fold A, dump per-inst ga_containers vs beam_N. ~1 DAgger+bench (~20min). Also single seed, no random_state → repeatability unknown. |
| Repro | GA (gen,idx) seed parallel reproducible; entropy flat → no mode collapse | `_eval_one` seeds `s=base_seed+gen*popsize+idx`, `random.seed(s)`+`np.random.seed(s%(2**32-1))` (loop.py:196-199); `base_seed` from `random.Random(seed).randint` (loop.py:283-284) → deterministic per (gen,idx). **BUT** serial `_fitness` path (loop.py:296-299) does NOT reseed → only the parallel path is reproducible; "結果不変" (result-invariant) is asserted in loop.py:530 docstring but **no A/B artifact comparing serial vs parallel exists**. ga_bench `_ga_worker` seeds `random.seed(BENCH_SEED+idx)` only — **np.random NOT seeded in ga_bench** (ga_bench.py:142-143), and `run_ga` internal RNG unknown → GA stochastic noise visible in progress.log (10.438⇄10.562). entropy: loop.py:333-339 logs decode_theta entropy per row; gen_summary mean_entropy 0.69-0.72 flat for 31-cat. | Partial. Seed *code* is present (settleable: parallel path reseeds, ga_bench does not seed numpy). "result-invariant vs what" is undocumented — needs a serial-vs-parallel identical-seed diff run to settle (~10min). Mode-collapse: entropy logged & flat (settleable from gen_summary). |

---

## 3. Factual Contradictions Found

1. **R0 slowdown "~6×" vs measured ~4×.** OVERNIGHT_GA_REPORT / brief say beam ~6× slower; progress.log shows beam wall 496.31s vs greedy 122.28s on the *same* 16-suite ≈ **4.06×**. (smoke_r0.py never ran on record; the 6× is a design-doc estimate, coevo_decoder.py:30 cites "~6x遅" as carried-over text.) Minor but a number mismatch.

2. **Brief's "reject cheap DAgger / strong-oracle-or-RL-only" conclusion is OUTDATED vs the checked-out tree.** Brief claim list (lines 88-90) stops at lite-DAgger degradation and concludes "棄却, 強 oracle / RL only". The HEAD commit `c1f8031` goes further and claims strong-oracle DAgger **already reached ceiling (0.0)**. The brief's "most-distrust target" framing and the actual repo state disagree about whether R2 ended in rejection or success.

3. **"dN≥1 率 17%→63%" (B claim) — the 17% baseline is step1_shaped's 15.6%**, not a separately reported figure; rounding presents 15.6%→17%. Recomputed step1_shaped r≥1 = 10/64 = 15.6%. Not a large discrepancy but the "17%" is not directly in any artifact.

4. **No surviving per-instance artifact for any Lane-3 headline.** Every R0/R1/R2 dN figure except the *final* Fold-B strong run exists only as a single aggregate line in `progress.log`; `last_bench.json` and `runs/r1/scorer.joblib` are overwritten in place. The "全 inst 悪化ゼロ" / "6/6 beam_ref 一致" / "val top1 unchanged" sub-claims have **zero primary artifact** — they are commit-message assertions only.

---

## 4. Eval-Set N and Metric Definitions (pinned)

- **Eval-set true N = 16.** `build_suite()` = 12 honban hard (`hard_01..12_g15_p12.json`, N_HARD=12 ga_bench.py:77,91) + 4 std (`case_{balanced,weight_bound,volume_bound,small_many}_seed42.json`, ga_bench.py:80-85; all 4 exist in `rui/datasets/`). baseline.json confirms `"instances": 16`.
- **Fold A eval N = 6** (`suite[0:6]` = hard_01..06). **Fold B eval N = 6** (`suite[10:16]` = hard_11, hard_12, + 4 std cases). Each 0.1667 dN unit ≈ exactly **1 instance out of 6** — i.e. "0.833 vs 1.33" = a 3-instance difference over 6; "0.333 vs 0.667" = a 2-instance difference over 6.
- **dN definition** (ga_bench.py:202-214): per instance `dN = ga_containers − beam_N`; `ga_containers` = `report["teacher_score_metrics"]["containers_used"]` from **real vanning_eval** `build_report` (ga_bench.py:160-166). `mean_dN` = arithmetic mean over instances whose `beam_N is not None` (ga_bench.py:213-214; instances with no beam ref are dropped from the mean — silent denominator change risk).
- **beam reference (the dN baseline)** = `beam_search_strong(items, beam_width=48, branch=12)` (antagonist.py:162-166, called with defaults in ga_bench `_beam_ref_worker` ga_bench.py:114). Pruning key `_partial_lex_key = (n_containers, mean_y_dev, dead_space)` (antagonist.py:39-46), final `_final_lex_key = (n, mean_dev)` (antagonist.py:49-53). It tries **4 item orderings** per dest group and takes the lex-best (antagonist.py:31-36,186-194). **It is a heuristic beam (width 48, branch 12, 4 seed orderings), NOT an exact/optimal lower bound** — dN is "GA vs strong-beam-heuristic gap", and is mechanically softened if beam's branch/width undershoots. Same pruning key (`_partial_lex_key`) is reused by the R1 teacher trace (r1_teacher.py:187) and the R2 lite oracle (r1_dagger.py:130) — i.e. teacher, oracle, and the dN yardstick all share one heuristic family (potential shared-bias / non-independent-reference concern).
- regret (Lane 2) = `dN + 1e-4·dDev` with dq_bonus=1e3 (regret.py:38-41); shaping `r + 0.5·(1−min_fill)` fed ONLY to CMA, pure r saved to trajectory & hard buffer (loop.py:341-343, 376-383).
- A/B run sizes: step1/step2 = 8gen×8pop = 64 evals; honban = 15gen×12pop = 180 evals (trajectory.csv line counts confirm).

## 5. Appendix — primary sources NOT readable / blocked

- pytest pass/fail counts: not run (brief rule 4 forbids); only `.pyc` collection evidence.
- Per-instance dN tables for R0/R1-logreg/R1-MLP-foldA/R2-lite/R2-strong-foldA: overwritten in `last_bench.json` (only Fold-B strong survives) and `runs/r1/scorer.joblib` (single overwritten model). progress.log preserves aggregate lines only.
- val-top1 numbers for logreg vs MLP: stdout only, not persisted.
- serial-vs-parallel "result invariant" comparison: no artifact.
