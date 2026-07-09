import subprocess
import time
import os
import sys
import json
import urllib.request

def kill_server():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(2)

def start_server(K):
    kill_server()
    
    # Overwrite server.log
    if os.path.exists("server.log"):
        os.remove("server.log")
        
    cmd = [
        "./build/bin/llama-server",
        "-m", "/home/kunweiz/Desktop/Ornith/weights/ornith-1.0-35b-Q6_K-MTP-final.gguf",
        "--jinja",
        "--port", "8080",
        "--host", "127.0.0.1",
        "--ctx-size", "262144",
        "-ngl", "999",
        "-ot", "exps=CPU",
        "--cache-type-k", "turbo4",
        "--cache-type-v", "turbo2",
        "-fa", "on",
        "-b", "3072",
        "-ub", "3072",
        "-t", "16",
        "-tb", "16",
        "--numa", "numactl"
    ]
    
    if K > 0:
        cmd.extend([
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(K),
            "--spec-draft-p-min", "0.0",
            "--spec-draft-n-min", "0",
            "--spec-draft-type-k", "turbo4",
            "--spec-draft-type-v", "turbo2",
            "--spec-draft-ngl", "0"
        ])
        
    env = os.environ.copy()
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    
    print(f"Starting server with K={K} (M={K+1})...")
    sys.stdout.flush()
    
    log_file = open("server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    
    # Wait for server to start listening
    for _ in range(60):
        time.sleep(1)
        if os.path.exists("server.log"):
            with open("server.log", "r", encoding="utf-8") as f:
                content = f.read()
            if "server is listening on" in content:
                print("Server is listening.")
                sys.stdout.flush()
                return proc
    print("Error: Server failed to start in 60s.")
    sys.stdout.flush()
    proc.terminate()
    sys.exit(1)

def run_request():
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "你是一个系统架构师。"},
            {"role": "user", "content": "详细列出投机解码的系统设计要点。"}
        ],
        "stream": True,
        "max_tokens": 500,
        "temperature": 0.0
    }
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"}, method="POST")
    
    print("Running generation request (500 tokens)...")
    sys.stdout.flush()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                pass # consume stream
        print("Generation finished.")
        sys.stdout.flush()
    except Exception as e:
        print(f"Request failed: {e}")
        sys.stdout.flush()

def extract_timings():
    # Wait for log writes
    time.sleep(2)
    with open("server.log", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    eval_time = 0.0
    calls = 0
    
    for line in lines:
        if "eval time =" in line:
            # e.g. 1.48.183.488 I slot print_timing: id  0 | task 0 |        eval time =   71084.69 ms
            m = re.search(r"eval time =\s+([\d.]+)\s+ms", line)
            if m:
                eval_time = float(m.group(1))
        if "statistics        draft-mtp:" in line:
            # e.g. statistics        draft-mtp: #calls(b,g,a) =    1    858    858
            m = re.search(r"#calls\(b,g,a\) =\s+\d+\s+(\d+)\s+\d+", line)
            if m:
                calls = int(m.group(1))
                
    if calls == 0 and eval_time > 0:
        # Standard decoding (K=0) has no draft-mtp statistics, calls = generated tokens
        for line in lines:
            if "print_timing: id  0 | task -1 | n_decoded =" in line:
                # e.g. print_timing: id  0 | task -1 | n_decoded =   2048
                m = re.search(r"n_decoded =\s+(\d+)", line)
                if m:
                    calls = int(m.group(1))
                    
    return eval_time, calls

import re
def main():
    results = {}
    
    # We test K = 0 (baseline), 1 (M=2), 2 (M=3), 3 (M=4)
    for K in [0, 1, 2, 3]:
        start_server(K)
        run_request()
        eval_time, calls = extract_timings()
        kill_server()
        
        if calls > 0:
            avg_latency = eval_time / calls
            results[K + 1] = {
                'eval_time': eval_time,
                'calls': calls,
                'avg_latency': avg_latency
            }
            print(f"Results for M={K+1}: eval_time={eval_time:.2f} ms, calls={calls}, avg_latency={avg_latency:.2f} ms")
        else:
            print(f"Results for M={K+1}: failed to parse timings from log")
        sys.stdout.flush()
        
    print("\n=== FINAL TG LATENCY BENCHMARK REPORT ===")
    print("| Batch Size (M) | Total Eval Time (ms) | Target Model Calls | Avg Latency per Call (ms) |")
    print("|----------------|----------------------|--------------------|---------------------------|")
    for M, res in sorted(results.items()):
        print(f"| {M} | {res['eval_time']:.2f} | {res['calls']} | {res['avg_latency']:.2f} |")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
