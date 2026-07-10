import json
from typing import List, Dict, Any, Sequence


def expected_acceptance(p: Sequence[float], k: int) -> float:
    e = 0.0
    cum = 1.0
    for i in range(0, k):
        cum *= p[i]
        e += cum
    return e


def expected_draft_length(
    p_base: Sequence[float], n_min: int, n_max: int, p_min: float,
) -> int:
    k = 0
    for i in range(0, n_max):
        if p_base[i] < p_min:
            break
        k += 1
    if k < n_min:
        return 0
    return k


def expected_time(
    n_max: int,
    p_min: float,
    n_min: int,
    t_tgt: Sequence[float],
    t_dft: float,
    p_base: Sequence[float],
) -> float:
    p = list(p_base)[:n_max]
    while len(p) < n_max:
        p.append(0.0)
    k = expected_draft_length(p, n_min, n_max, p_min)
    a = expected_acceptance(p, k) if k >= n_min else 0.0
    m = max(0, min(k, len(t_tgt) - 1))
    time = k * t_dft + t_tgt[m]
    return time / (a + 1.0)


def expected_speedup(
    n_max: int,
    p_min: float,
    n_min: int,
    t_tgt: Sequence[float],
    t_dft: float,
    p_base: Sequence[float],
) -> float:
    if t_tgt[0] <= 0.0:
        return 0.0
    return t_tgt[0] / expected_time(n_max, p_min, n_min, t_tgt, t_dft, p_base)


def grid_search(
    n_max_range: Sequence[int],
    p_min_range: Sequence[float],
    n_min_range: Sequence[int],
    t_tgt: Sequence[float],
    t_dft: float,
    p_base: Sequence[float],
) -> List[Dict[str, Any]]:
    rows = []
    for n_max in n_max_range:
        for p_min in p_min_range:
            for n_min in n_min_range:
                if n_min > n_max:
                    continue
                t = expected_time(n_max, p_min, n_min, t_tgt, t_dft, p_base)
                s = expected_speedup(
                    n_max, p_min, n_min, t_tgt, t_dft, p_base)
                rows.append({
                    "n_max": n_max,
                    "p_min": p_min,
                    "n_min": n_min,
                    "expected_time_ms": t,
                    "expected_speedup": s,
                })
    return sorted(rows, key=lambda x: (x["expected_time_ms"], -x["n_min"]))


def main():
    t_tgt = [44.66, 79.15, 83.53, 100.64]
    t_dft = 1.50
    p_base = [0.85, 0.65, 0.40]
    rows = grid_search(
        range(1, 5),
        [0.05, 0.10, 0.15, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        [0, 1, 2, 3],
        t_tgt, t_dft, p_base,
    )
    best = rows[0]
    print(json.dumps(best, indent=2))
    print("\nTop 10 configurations:")
    for r in rows[:10]:
        print(
            f"  n_max={r['n_max']} p_min={r['p_min']} n_min={r['n_min']} "
            f"time={r['expected_time_ms']:.2f}ms "
            f"speedup={r['expected_speedup']:.3f}x")


if __name__ == "__main__":
    main()
