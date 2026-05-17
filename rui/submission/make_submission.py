"""提出用 layout_output.json 生成 (レーン3 強DAgger 学習デコーダ)。

`submission/items_input.json` (rinseishisa/Vanning-layout-algorithm:main の
オーソリティ版) を run_ga + pack_items_learned (= 強oracle DAgger で全16inst
訓練した最強モデル) で vanning し、**実 vanning_eval で検証**してから
`submission/layout_output.json` を書く。

盲目採用を避けるため greedy も同条件で走らせ並べて表示する
(learned が disq か greedy より悪ければ採用しない判断材料)。
team_name はユーザー決定により "rui" で確定 (本パイプラインからの
企業/授業提出は行わない方針 — 2026-05-18)。
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # rui/submission/ -> repo root
for p in (str(_REPO_ROOT),
          r"G:\マイドライブ\ideamap\worksp\vanning-eval\vanning_eval_rui\src"):
    if p not in sys.path:
        sys.path.insert(0, p)

from rui.adv_lane.coevo_decoder import pack_items_learned  # noqa: E402
from rui.adv_lane.loop import _make_dataframe  # noqa: E402
from rui.algorithm_a import (  # noqa: E402
    build_items,
    build_output_json,
    pack_items,
    run_ga,
)

_HERE = Path(__file__).parent
INPUT_PATH = _HERE / "items_input.json"
OUTPUT_PATH = _HERE / "layout_output.json"

# メタ — team_name はユーザー決定で "rui" 確定 (提出予定なし、2026-05-18)
TEAM_NAME = "rui"
GA_GEN = 20
GA_POP = 14
SEED = 1234


def _load_items(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return build_items(_make_dataframe(data))


def _validate(layout: dict, input_path: Path) -> dict:
    """実 vanning_eval で検証 (中間経路でなく本物を叩く)。"""
    from vanning_eval.report import build_report
    from vanning_eval.schema import load_items, load_layout

    tmp = _HERE / "_tmp_validate.json"
    tmp.write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
    report = build_report(load_layout(tmp), load_items(input_path))
    tmp.unlink(missing_ok=True)
    t = report["teacher_score_metrics"]
    devs = t.get("cog_dev_per_container") or [0.0]
    return {
        "verdict": report["verdict"],
        "containers": int(t["containers_used"]),
        "mean_cog_dev": round(sum(devs) / len(devs), 1) if devs else 0.0,
        "n_violations": len(report["disqualifications"]),
        "disqualifications": report["disqualifications"][:5],
    }


def _run(decoder, label: str):
    import random

    random.seed(SEED)
    items = _load_items(INPUT_PATH)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        containers, _ = run_ga(items, generations=GA_GEN, pop_size=GA_POP,
                               decoder=decoder)
        ms = int((time.perf_counter() - t0) * 1000)
        layout = build_output_json(containers, TEAM_NAME, ms)
    wall = time.perf_counter() - t0
    v = _validate(layout, INPUT_PATH)
    print(f"[{label}] verdict={v['verdict']} containers={v['containers']} "
          f"mean_cog_dev={v['mean_cog_dev']} viol={v['n_violations']} "
          f"wall={wall:.1f}s")
    if v["disqualifications"]:
        print(f"  DQ: {v['disqualifications']}")
    return layout, v


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    print(f"input: {INPUT_PATH.name} / team_name={TEAM_NAME!r} (確定値)")
    g_layout, g_v = _run(pack_items, "greedy")
    l_layout, l_v = _run(pack_items_learned, "learned(強DAgger)")

    # 両方を submission に保存 (どちらを提出するかはユーザー判断)
    (_HERE / "layout_output_learned.json").write_text(
        json.dumps(l_layout, ensure_ascii=False, indent=2), encoding="utf-8")
    (_HERE / "layout_output_greedy.json").write_text(
        json.dumps(g_layout, ensure_ascii=False, indent=2), encoding="utf-8")

    def _lex(v):  # 辞書式キー (小さいほど上位): (disq, N, cog, ms 略)
        return (0 if v["verdict"] == "pass" else 1,
                v["containers"], v["mean_cog_dev"])

    better = "greedy" if _lex(g_v) <= _lex(l_v) else "learned"
    chosen = g_layout if better == "greedy" else l_layout
    OUTPUT_PATH.write_text(
        json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n辞書式(①N ②COGズレ ③時間) 上位 = {better}")
    print(f"  greedy : N={g_v['containers']} cog={g_v['mean_cog_dev']} "
          f"verdict={g_v['verdict']}")
    print(f"  learned: N={l_v['containers']} cog={l_v['mean_cog_dev']} "
          f"verdict={l_v['verdict']}")
    print(f"wrote {OUTPUT_PATH.name} = {better} (辞書式上位を提出既定に) "
          f"+ layout_output_learned.json / layout_output_greedy.json 併置")
    print(f"team_name={TEAM_NAME!r} (確定 / 本パイプラインからの提出はしない方針)")


if __name__ == "__main__":
    main()
