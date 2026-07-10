import argparse
import json
import os
import re
import subprocess
import time
from typing import Dict, Any


def run_llama_cli(
    model: str,
    prompt: str,
    n_max: int,
    p_min: float,
    n_min: int,
    max_tokens: int,
    threads: int = 16,
    batch: int = 3072,
    ubatch: int = 3072,
    ctx_size: int = 262144,
    cores: str = "0-15",
    log_path: str = "titrate.log",
) -> None:
    cmd = [
        "numactl", "--interleave=all",
        "taskset", "-c", cores,
        "./build/bin/llama-cli",
        "-m", model,
        "--ctx-size", str(ctx_size),
        "-ngl", "999",
        "-ot", "exps=CPU",
        "--cache-type-k", "turbo4",
        "--cache-type-v", "turbo2",
        "-fa", "on",
        "-b", str(batch),
        "-ub", str(ubatch),
        "-t", str(threads),
        "-tb", str(threads),
        "--numa", "numactl",
        "--no-display-prompt",
        "--reasoning", "off",
        "-n", str(max_tokens),
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(n_max),
        "--spec-draft-p-min", str(p_min),
        "--spec-draft-n-min", str(n_min),
        "--spec-draft-type-k", "turbo4",
        "--spec-draft-type-v", "turbo2",
        "--spec-draft-ngl", "0",
        "-p", prompt,
    ]
    env = os.environ.copy()
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)


def parse_log(path: str) -> Dict[str, Any]:
    stats = {
        "eval_time": 0.0, "eval_tokens": 0,
        "prompt_eval_time": 0.0, "prompt_eval_tokens": 0,
        "spec_type": "none",
        "calls_generated": 0, "calls_accepted": 0,
        "generated_tokens": 0, "accepted_tokens": 0,
    }
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "prompt eval time =" in line:
                m = re.search(r"prompt eval time =\s+([\d.]+)\s+ms", line)
                if m:
                    stats["prompt_eval_time"] = float(m.group(1))
                m_tok = re.search(r"/\s+(\d+)\s+tokens", line)
                if m_tok:
                    stats["prompt_eval_tokens"] = int(m_tok.group(1))
            elif "eval time =" in line:
                m = re.search(r"eval time =\s+([\d.]+)\s+ms", line)
                if m:
                    stats["eval_time"] = float(m.group(1))
                m_tok = re.search(r"/\s+(\d+)\s+tokens", line)
                if m_tok:
                    stats["eval_tokens"] = int(m_tok.group(1))
            elif "statistics" in line and "#calls(b,g,a)" in line:
                m = re.search(
                    r"statistics\s+([^\s:]+):\s+#calls\(b,g,a\)\s+=\s+"
                    r"(\d+)\s+(\d+)\s+(\d+),\s*#gen drafts\s+=\s+(\d+),\s*"
                    r"#acc drafts\s+=\s+(\d+),\s*#gen tokens\s+=\s+(\d+),\s*"
                    r"#acc tokens\s+=\s+(\d+)",
                    line,
                )
                if m:
                    stats["spec_type"] = m.group(1)
                    stats["calls_generated"] = int(m.group(3))
                    stats["calls_accepted"] = int(m.group(4))
                    stats["generated_tokens"] = int(m.group(7))
                    stats["accepted_tokens"] = int(m.group(8))
    return stats


def compute_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(stats)
    m["acceptance_rate"] = (
        stats["accepted_tokens"] / stats["generated_tokens"]
        if stats["generated_tokens"] else 0.0
    )
    m["tokens_per_second"] = (
        1000.0 * stats["eval_tokens"] / stats["eval_time"]
        if stats["eval_time"] else 0.0
    )
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Detailed design of speculative decoding.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n-max-range", default="0,1,2,3")
    parser.add_argument("--p-min-range", default="0.05,0.1,0.16,0.2,0.3,0.4")
    parser.add_argument("--n-min-range", default="0,1,2")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch", type=int, default=3072)
    parser.add_argument("--ubatch", type=int, default=3072)
    parser.add_argument("--ctx-size", type=int, default=262144)
    parser.add_argument("--cores", default="0-15")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", default="titrate_results.jsonl")
    args = parser.parse_args()

    n_maxs = [int(x) for x in args.n_max_range.split(",") if x]
    p_mins = [float(x) for x in args.p_min_range.split(",") if x]
    n_mins = [int(x) for x in args.n_min_range.split(",") if x]

    with open(args.output, "w", encoding="utf-8") as out:
        for n_max in n_maxs:
            for p_min in p_mins:
                for n_min in n_mins:
                    if n_min > n_max:
                        continue
                    for t in range(args.trials):
                        log = f"titrate_n{n_max}_p{p_min}_m{n_min}_t{t}.log"
                        run_llama_cli(
                            args.model, args.prompt, n_max, p_min, n_min,
                            args.max_tokens, args.threads, args.batch,
                            args.ubatch, args.ctx_size, args.cores, log,
                        )
                        stats = parse_log(log)
                        metrics = compute_metrics(stats)
                        row = {
                            "n_max": n_max,
                            "p_min": p_min,
                            "n_min": n_min,
                            "trial": t,
                            "timestamp": time.time(),
                            **metrics,
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
