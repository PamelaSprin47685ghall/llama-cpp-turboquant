#!/usr/bin/env python3
"""
Ornith 35B Ablation Benchmark — Precise per-component latency profiling.

Design principles:
  1. Single variable changed per experiment
  2. Fixed prompt + temp=0 for reproducibility
  3. Parse ACTUAL log timings (not derived/estimated)
  4. Warm up with 10 tokens, then measure next 200
  5. Each experiment: start server, warmup, measure, kill
"""

import subprocess, time, os, sys, json, urllib.request, re, signal

DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(DIR)
SERVER = os.path.join(REPO, "build", "bin", "llama-server")
MODEL = "/home/kunweiz/Desktop/Ornith/weights/ornith-1.0-35b-Q6_K-MTP-final.gguf"
LOG = os.path.join(REPO, "server.log")

COMMON = [
    SERVER,
    "-m", MODEL,
    "--jinja",
    "--port", "8080",
    "--host", "127.0.0.1",
    "-ngl", "999",
    "-fa", "on",
    "-b", "3072",
    "-ub", "3072",
    "--numa", "numactl",
]

PROMPT = [
    {"role": "system", "content": "你是一个系统架构师。"},
    {"role": "user", "content": "详细列出投机解码的系统设计要点，包括数学模型。"},
]

def kill_server():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

def start_server(extra_args, env_extra=None, taskset=None, numactl=True):
    kill_server()
    if os.path.exists(LOG):
        os.remove(LOG)

    cmd = []
    if numactl:
        cmd += ["numactl", "--interleave=all"]
    if taskset:
        cmd += ["taskset", "-c", taskset]
    cmd += COMMON + extra_args

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    log_f = open(LOG, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)

    for _ in range(120):
        time.sleep(1)
        if os.path.exists(LOG):
            with open(LOG) as f:
                if "server is listening" in f.read():
                    return proc
    proc.kill()
    return None

def warmup_then_measure(n_warmup=10, n_measure=200):
    """Send two requests: short warmup, then measured run."""
    url = "http://127.0.0.1:8080/v1/chat/completions"

    # Warmup
    payload = {"messages": PROMPT, "stream": False, "max_tokens": n_warmup, "temperature": 0.0}
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()
    except Exception as e:
        return None, f"warmup failed: {e}"

    # Clear log to get clean measurement
    with open(LOG, "w") as f:
        pass

    # Measure
    payload["max_tokens"] = n_measure
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
    except Exception as e:
        return None, f"measure failed: {e}"

    # Parse log
    return parse_log(), None

def parse_log():
    """Parse actual server timings from log."""
    with open(LOG) as f:
        lines = f.readlines()

    result = {
        "tg_ms_per_tok": None,
        "tg_tps": None,
        "n_decoded": None,
        "graphs_reused": None,
        "draft_calls_b": None,
        "draft_calls_g": None,
        "draft_calls_a": None,
        "pp_ms_per_tok": None,
        "pp_tps": None,
        "pp_n_tokens": None,
    }

    for line in lines:
        m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second", line)
        if m:
            result["tg_ms_per_tok"] = float(m.group(3))
            result["tg_tps"] = float(m.group(4))
            result["n_decoded"] = int(m.group(2))

        m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second", line)
        if m:
            result["pp_ms_per_tok"] = float(m.group(3))
            result["pp_tps"] = float(m.group(4))
            result["pp_n_tokens"] = int(m.group(2))

        m = re.search(r"graphs reused\s*=\s*(\d+)", line)
        if m:
            result["graphs_reused"] = int(m.group(1))

        m = re.search(r"#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            result["draft_calls_b"] = int(m.group(1))
            result["draft_calls_g"] = int(m.group(2))
            result["draft_calls_a"] = int(m.group(3))

    return result

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================
# Each: (id, description, extra_args, env_extra, taskset, numactl)

EXPERIMENTS = [
    # ---- A series: locate the bottleneck ----
    ("A1a", "t=8 P-core only, ctx=262k, no mmap, graphs off",
     ["--ctx-size", "262144", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8", "--no-mmap"],
     {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "0-15", True),

    ("A1b", "t=16 (HT), ctx=262k, no mmap, graphs off",
     ["--ctx-size", "262144", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "16", "-tb", "16", "--no-mmap"],
     {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "0-15", True),

    ("A3a", "t=8 P-core, ctx=8192, no mmap, graphs off",
     ["--ctx-size", "8192", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8", "--no-mmap"],
     {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "0-15", True),

    ("A3b", "t=8 P-core, ctx=8192, no mmap, graphs ON",
     ["--ctx-size", "8192", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8", "--no-mmap"],
     {}, "0-15", True),

    ("A4a", "t=8 P-core, ctx=262k, WITH mmap (default), graphs off",
     ["--ctx-size", "262144", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8"],
     {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "0-15", True),

    ("A7a", "t=8 P-core, ctx=262k, no mmap, graphs ON",
     ["--ctx-size", "262144", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8", "--no-mmap"],
     {}, "0-15", True),

    # ---- B series: spec decode latency at clean ctx ----
    ("B1_k1", "K=1 spec, t=8, ctx=8192, no mmap, graphs off",
     ["--ctx-size", "8192", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8", "--no-mmap",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
      "--spec-draft-p-min", "0.0", "--spec-draft-n-min", "0",
      "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo2",
      "--spec-draft-ngl", "0"],
     {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "0-15", True),

    ("B1_k2", "K=2 spec, t=8, ctx=8192, no mmap, graphs off",
     ["--ctx-size", "8192", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8", "--no-mmap",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
      "--spec-draft-p-min", "0.0", "--spec-draft-n-min", "0",
      "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo2",
      "--spec-draft-ngl", "0"],
     {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "0-15", True),

    ("B1_k3", "K=3 spec, t=8, ctx=8192, no mmap, graphs off",
     ["--ctx-size", "8192", "-ot", "exps=CPU", "--cache-type-k", "turbo4",
      "--cache-type-v", "turbo2", "-t", "8", "-tb", "8", "--no-mmap",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
      "--spec-draft-p-min", "0.0", "--spec-draft-n-min", "0",
      "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo2",
      "--spec-draft-ngl", "0"],
     {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "0-15", True),
]

def main():
    # Allow selecting experiments via CLI args
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        experiments = [(e[0],) + e[1:] for e in EXPERIMENTS if e[0] in wanted]
    else:
        experiments = EXPERIMENTS

    results = []
    for exp_id, desc, extra_args, env_extra, taskset, numactl in experiments:
        print(f"\n{'='*60}")
        print(f"[{exp_id}] {desc}")
        print(f"{'='*60}")
        sys.stdout.flush()

        proc = start_server(extra_args, env_extra, taskset, numactl)
        if proc is None:
            print(f"  FAILED: server didn't start")
            results.append((exp_id, desc, None, "server start failed"))
            continue

        timing, err = warmup_then_measure()
        if err:
            print(f"  FAILED: {err}")
            results.append((exp_id, desc, None, err))
        else:
            print(f"  TG: {timing['tg_ms_per_tok']:.2f} ms/tok ({timing['tg_tps']:.2f} t/s), n={timing['n_decoded']}")
            if timing.get('pp_tps'):
                print(f"  PP: {timing['pp_ms_per_tok']:.2f} ms/tok ({timing['pp_tps']:.2f} t/s), n={timing['pp_n_tokens']}")
            if timing.get('draft_calls_a') is not None:
                b, g, a = timing['draft_calls_b'], timing['draft_calls_g'], timing['draft_calls_a']
                acc_rate = a / g * 100 if g > 0 else 0
                print(f"  Draft: b={b}, g={g}, a={a}, accept_rate={acc_rate:.1f}%")
            results.append((exp_id, desc, timing, None))

        kill_server()
        sys.stdout.flush()

    # Summary
    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'ID':<10} {'ms/tok':>8} {'t/s':>8} {'accept%':>8}  Description")
    print("-" * 70)
    for exp_id, desc, timing, err in results:
        if timing:
            ms = timing.get('tg_ms_per_tok', 0)
            tps = timing.get('tg_tps', 0)
            acc = ""
            if timing.get('draft_calls_a') is not None and timing['draft_calls_g']:
                acc = f"{timing['draft_calls_a']/timing['draft_calls_g']*100:.1f}"
            print(f"{exp_id:<10} {ms:>8.2f} {tps:>8.2f} {acc:>8}  {desc}")
        else:
            print(f"{exp_id:<10} {'ERR':>8} {'':>8} {'':>8}  {desc} ({err})")

if __name__ == "__main__":
    main()
