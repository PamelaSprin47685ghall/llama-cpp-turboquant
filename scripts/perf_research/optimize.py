import argparse
import json
from typing import List, Dict, Any, Sequence

from spec_model import expected_time, expected_speedup, expected_acceptance, expected_draft_length


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = {}
    for r in rows:
        key = (r.get("n_max"), r.get("p_min"), r.get("n_min"))
        grouped.setdefault(key, []).append(r)
    out = []
    for key, g in grouped.items():
        t = sum(r["tokens_per_second"] for r in g) / len(g)
        out.append({
            "n_max": key[0],
            "p_min": key[1],
            "n_min": key[2],
            "tokens_per_second": t,
            "samples": len(g),
        })
    return out


def fit_t_tgt(
    rows: List[Dict[str, Any]],
    p_base: Sequence[float],
    t_dft: float,
) -> List[float]:
    """Recover T_tgt(M) from measured net throughput.

    tokens_per_second is net throughput = Time / (a + 1), so:
      Time = (1000 / tps) * (a + 1)
    and Time = k * t_dft + T_tgt(k), therefore:
      T_tgt(k) = (1000 / tps) * (a + 1) - k * t_dft

    For n_max = 0: k = 0, a = 0 → T_tgt(0) = 1000 / tps.
    For n_max > 0: use rows where all draft positions are generated
    (p_min <= p_base[n_max - 1] and n_min <= n_max), then
    k = n_max, a = expected_acceptance(p_base, n_max).
    """
    accum = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for r in rows:
        n = r.get("n_max", 0)
        tps = r.get("tokens_per_second", 0.0)
        if n not in accum or tps <= 0.0:
            continue

        if n == 0:
            accum[0] += 1000.0 / tps
            counts[0] += 1
            continue

        pm = r.get("p_min", 1.0)
        nm = r.get("n_min", 0)
        # Only use rows where all n_max draft positions are generated
        if pm <= p_base[n - 1] and nm <= n:
            k = n
            a = expected_acceptance(p_base, k)
            total_time = (1000.0 / tps) * (a + 1.0)
            accum[n] += total_time - k * t_dft
            counts[n] += 1

    # Fallback: if no row matched the strict filter for n > 0,
    # use the row with smallest p_min and apply the same inversion
    for n in (1, 2, 3):
        if counts[n] > 0:
            continue
        candidates = [
            r for r in rows
            if r.get("n_max") == n
            and r.get("tokens_per_second", 0.0) > 0.0
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: r.get("p_min", 1.0))
        pm = best.get("p_min", 1.0)
        nm = best.get("n_min", 0)
        k = expected_draft_length(p_base, nm, n, pm)
        a = expected_acceptance(p_base, k) if k >= nm else 0.0
        total_time = (1000.0 / best["tokens_per_second"]) * (a + 1.0)
        accum[n] = total_time - k * t_dft
        counts[n] = 1

    return [accum[i] / counts[i] if counts[i] else 0.0 for i in range(4)]


def fit_p_base(rows: List[Dict[str, Any]]) -> List[float]:
    """Estimate per-position acceptance probabilities p1, p2, p3.

    For n_max=K with p_min=0 (all draft tokens verified):
      E[generated] = 1 + p1 + p1*p2 + ... + p1*...*p_{K-1}
      E[accepted]  = p1 + p1*p2 + ... + p1*...*p_K
      r_K = accepted_tokens / generated_tokens

    Solve sequentially:
      p1 = r_1
      p2 from r_2 = (p1 + p1*p2) / (1 + p1)
      p3 from r_3 = (p1 + p1*p2 + p1*p2*p3) / (1 + p1 + p1*p2)
    """
    # Group by n_max, pick rows with smallest p_min (prefer 0.0)
    by_nmax: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    for r in rows:
        n = r.get("n_max", 0)
        if n in by_nmax and r.get("generated_tokens", 0) > 0:
            by_nmax[n].append(r)

    # For each n_max, select the group with the lowest p_min
    ratios: Dict[int, float] = {}
    for n in (1, 2, 3):
        if not by_nmax[n]:
            continue
        by_pmin: Dict[float, List[Dict[str, Any]]] = {}
        for r in by_nmax[n]:
            pm = r.get("p_min", 1.0)
            by_pmin.setdefault(pm, []).append(r)
        best_pmin = min(by_pmin.keys())
        selected = by_pmin[best_pmin]
        total_acc = sum(r.get("accepted_tokens", 0) for r in selected)
        total_gen = sum(r.get("generated_tokens", 0) for r in selected)
        if total_gen > 0:
            ratios[n] = total_acc / total_gen

    p = [0.0, 0.0, 0.0]

    # p1 = r_1
    if 1 in ratios:
        p[0] = max(0.0, min(1.0, ratios[1]))

    # r_2 = (p1 + p1*p2) / (1 + p1)  =>  p2 = (r_2*(1+p1) - p1) / p1
    if 2 in ratios and p[0] > 0.0:
        r2 = ratios[2]
        p[1] = (r2 * (1.0 + p[0]) - p[0]) / p[0]
        p[1] = max(0.0, min(1.0, p[1]))

    # r_3 = (p1 + p1*p2 + p1*p2*p3) / (1 + p1 + p1*p2)
    # => p3 = (r_3*(1+p1+p1*p2) - p1 - p1*p2) / (p1*p2)
    if 3 in ratios and p[0] > 0.0 and p[1] > 0.0:
        r3 = ratios[3]
        denom = p[0] * p[1]
        numer = r3 * (1.0 + p[0] + denom) - p[0] - denom
        p[2] = numer / denom
        p[2] = max(0.0, min(1.0, p[2]))

    return p


def optimize(
    rows: List[Dict[str, Any]],
    t_dft: float,
    p_base: Sequence[float],
) -> Dict[str, Any]:
    t_tgt = fit_t_tgt(rows, p_base, t_dft)
    candidates = []
    for n_max in range(1, 5):
        for p_min in [0.05, 0.10, 0.15, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            for n_min in range(0, n_max + 1):
                t = expected_time(n_max, p_min, n_min, t_tgt, t_dft, p_base)
                candidates.append({
                    "n_max": n_max,
                    "p_min": p_min,
                    "n_min": n_min,
                    "expected_time_ms": t,
                    "expected_speedup": expected_speedup(
                        n_max, p_min, n_min, t_tgt, t_dft, p_base),
                    "t_tgt": t_tgt,
                    "p_base": list(p_base),
                })
    candidates.sort(key=lambda x: (x["expected_time_ms"], -x["n_min"]))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="titrate_results.jsonl")
    parser.add_argument("--t-dft", type=float, default=1.5)
    parser.add_argument("--output", default="optimal_config.json")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    agg = aggregate(rows)
    p_base = fit_p_base(rows)
    best = optimize(agg, args.t_dft, p_base)

    print(json.dumps(best, indent=2, ensure_ascii=False))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"Saved optimal config to {args.output}")


if __name__ == "__main__":
    main()
