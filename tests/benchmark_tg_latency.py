import subprocess
import time
import os
import re
import sys
import json
import urllib.request
import argparse
from typing import Dict, Any


DEFAULT_PROMPT = "\u4f60\u662f\u4e00\u4e2a\u7cfb\u7edf\u67b6\u6784\u5e08\u3002"
USER_PROMPT = "\u8be6\u7ec6\u5217\u51fa\u6295\u673a\u89e3\u7801\u7684\u7cfb\u7edf\u8bbe\u8ba1\u8981\u70b9\u3002"


def parse_log(path: str) -> Dict[str, Any]:
    stats = {
        "eval_time": 0.0, "eval_tokens": 0,
        "prompt_eval_time": 0.0, "prompt_eval_tokens": 0,
        "spec_type": "none",
        "calls_begin": 0, "calls_generated": 0, "calls_accepted": 0,
        "generated_drafts": 0, "accepted_drafts": 0,
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
                    stats["calls_begin"] = int(m.group(2))
                    stats["calls_generated"] = int(m.group(3))
                    stats["calls_accepted"] = int(m.group(4))
                    stats["generated_drafts"] = int(m.group(5))
                    stats["accepted_drafts"] = int(m.group(6))
                    stats["generated_tokens"] = int(m.group(7))
                    stats["accepted_tokens"] = int(m.group(8))
    return stats


def compute_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(stats)
    m["acceptance_rate"] = (
        stats["accepted_tokens"] / stats["generated_tokens"]
        if stats["generated_tokens"] else 0.0
    )
    m["avg_latency_per_eval_token"] = (
        stats["eval_time"] / stats["eval_tokens"]
        if stats["eval_tokens"] else 0.0
    )
    m["avg_latency_per_prompt_token"] = (
        stats["prompt_eval_time"] / stats["prompt_eval_tokens"]
        if stats["prompt_eval_tokens"] else 0.0
    )
    m["tokens_per_second"] = (
        1000.0 * stats["eval_tokens"] / stats["eval_time"]
        if stats["eval_time"] else 0.0
    )
    return m


def kill_server():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(2)


def start_server(
    K: int,
    model: str,
    port: int = 8080,
    ctx_size: int = 262144,
    threads: int = 16,
    batch: int = 3072,
    ubatch: int = 3072,
    p_min: float = 0.0,
    n_min: int = 0,
    cores: str = "0-15",
) -> subprocess.Popen:
    kill_server()
    if os.path.exists("server.log"):
        os.remove("server.log")
    cmd = [
        "numactl", "--interleave=all",
        "taskset", "-c", cores,
        "./build/bin/llama-server",
        "-m", model,
        "--jinja",
        "--port", str(port),
        "--host", "127.0.0.1",
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
    ]
    if K > 0:
        cmd.extend([
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(K),
            "--spec-draft-p-min", str(p_min),
            "--spec-draft-n-min", str(n_min),
            "--spec-draft-type-k", "turbo4",
            "--spec-draft-type-v", "turbo2",
            "--spec-draft-ngl", "0",
        ])
    env = os.environ.copy()
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    print(f"Starting server with K={K} (M={K+1}) p_min={p_min} n_min={n_min}...")
    sys.stdout.flush()
    log_file = open("server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    for _ in range(300):
        time.sleep(1)
        if os.path.exists("server.log"):
            with open("server.log", "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "server is listening on" in content:
                print("Server is listening.")
                sys.stdout.flush()
                return proc
    print("Error: Server failed to start in 300s.")
    sys.stdout.flush()
    proc.terminate()
    sys.exit(1)


def run_request(port: int, max_tokens: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": DEFAULT_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=req_data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    print(f"Running generation request ({max_tokens} tokens)...")
    sys.stdout.flush()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                pass
        print("Generation finished.")
        sys.stdout.flush()
    except Exception as e:
        print(f"Request failed: {e}")
        sys.stdout.flush()


def run_one(K: int, args: argparse.Namespace) -> Dict[str, Any]:
    proc = start_server(
        K, args.model, args.port, args.ctx_size, args.threads,
        args.batch, args.ubatch, args.p_min, args.n_min, args.cores,
    )
    run_request(args.port, args.max_tokens)
    time.sleep(2)
    stats = parse_log("server.log")
    kill_server()
    return compute_metrics(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TG latency for MTP speculative decoding.")
    parser.add_argument("--model", default="/home/kunweiz/Desktop/Ornith/weights/ornith-1.0-35b-IQ4NL-MTP-final.gguf")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=262144)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch", type=int, default=3072)
    parser.add_argument("--ubatch", type=int, default=3072)
    parser.add_argument("--cores", default="0-15")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--ks", default="0,1,2,3")
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    results = {}
    for K in [int(k) for k in args.ks.split(",") if k]:
        results[K + 1] = run_one(K, args)
        r = results[K + 1]
        print(
            f"M={K+1}: eval={r['eval_time']:.2f}ms/{r['eval_tokens']}t "
            f"accept={r['acceptance_rate']:.3f} "
            f"t/s={r['tokens_per_second']:.2f}"
        )
        sys.stdout.flush()

    print("\n=== FINAL TG LATENCY BENCHMARK REPORT ===")
    print("| M | eval_time | eval_tokens | prompt_time | prompt_tokens | "
          "accept_rate | t/s |")
    print("|---|-----------|-------------|-------------|---------------|"
          "-------------|-----|")
    for M in sorted(results):
        r = results[M]
        print(f"| {M} | {r['eval_time']:.2f} | {r['eval_tokens']} | "
              f"{r['prompt_eval_time']:.2f} | {r['prompt_eval_tokens']} | "
              f"{r['acceptance_rate']:.3f} | {r['tokens_per_second']:.2f} |")
    sys.stdout.flush()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {args.output}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
