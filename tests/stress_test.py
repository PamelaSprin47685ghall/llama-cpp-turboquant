#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import time
import json
import urllib.request
import urllib.error

def parse_args():
    p = argparse.ArgumentParser(description="万象阵 (DKVT) 自动化集成压测脚本")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--read-only", action="store_true")
    p.add_argument("--input-file", default="/home/kunweiz/Desktop/vibe/wanxiangzhen/PRD/PRD.md")
    p.add_argument("--loop-count", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=100)
    return p.parse_args()

def load_prompt(file_path, loop_count):
    if not os.path.exists(file_path):
        print(f"[错误] 输入文件不存在: {file_path}", file=sys.stderr)
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    prompt = content * loop_count if loop_count > 1 else content
    print(f"[准备] 输入字符数: {len(prompt)}")
    return prompt

def parse_sse_line(line_str, state):
    if not line_str.startswith("data:"):
        return True
    data_json = line_str[5:].strip()
    if data_json == "[DONE]":
        return False
    try:
        chunk = json.loads(data_json)
        state["chunks"] += 1
        if state["ttft_time"] is None and "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            if "delta" in choice and ("content" in choice["delta"] or choice["delta"]):
                state["ttft_time"] = time.time()
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
            state["prompt_tokens"] = usage.get("prompt_tokens", 0)
            state["completion_tokens"] = usage.get("completion_tokens", 0)
    except json.JSONDecodeError:
        pass
    return True

def run_single_request(url, prompt, read_only, max_tokens):
    payload = {
        "messages": [
            {"role": "system", "content": "你是一个专业的系统架构师。请阅读下面的万象阵 PRD 文档。"},
            {"role": "user", "content": f"以下是项目文档：\n\n{prompt}\n\n请详细分析这份万象阵 PRD 的利弊，并给出深入的设计建议。"}
        ],
        "stream": True,
        "max_tokens": 1 if read_only else max_tokens,
        "temperature": 0.7,
        "stream_options": {"include_usage": True}
    }
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    state = {"chunks": 0, "ttft_time": None, "prompt_tokens": 0, "completion_tokens": 0}
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            for line in response:
                line_str = line.decode("utf-8").strip()
                if line_str and not parse_sse_line(line_str, state):
                    break
    except Exception as e:
        print(f"[异常] 请求失败: {e}", file=sys.stderr)
        return None
    end = time.time()
    return calculate_metrics(start, end, state, prompt)

def calculate_metrics(start, end, state, prompt):
    ttft_val = (state["ttft_time"] - start) if state["ttft_time"] else (end - start)
    total_duration = end - start
    prompt_tokens = state["prompt_tokens"] if state["prompt_tokens"] > 0 else int(len(prompt) * 0.4)
    completion_tokens = state["completion_tokens"] if state["completion_tokens"] > 0 else state["chunks"]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft": ttft_val,
        "total_duration": total_duration,
        "prefill_speed": prompt_tokens / ttft_val if ttft_val > 0 else 0,
        "decode_speed": completion_tokens / (total_duration - ttft_val) if (total_duration - ttft_val) > 0 else 0
    }

def print_result(res):
    print(f"[完成] 压测结果:")
    print(f"  - 输入 Token 数 (估计/实际): {res['prompt_tokens']}")
    print(f"  - 输出 Token 数 (实际): {res['completion_tokens']}")
    print(f"  - 首包延迟 (TTFT): {res['ttft']:.3f} 秒")
    print(f"  - 实际 Prefill 速度: {res['prefill_speed']:.2f} t/s")
    print(f"  - 实际 Token 产生速度: {res['decode_speed']:.2f} t/s")
    print(f"  - 总耗时: {res['total_duration']:.3f} 秒")

def run_concurrency(url, prompt, args):
    import threading
    results = [None] * args.concurrency
    threads = [threading.Thread(target=lambda i=i: results.__setitem__(i, run_single_request(url, prompt, args.read_only, args.max_tokens))) for i in range(args.concurrency)]
    for t in threads: t.start()
    for t in threads: t.join()
    valid = [r for r in results if r]
    if not valid:
        print("[失败] 所有并发请求均失败", file=sys.stderr)
        return []
    avg_ttft = sum(r['ttft'] for r in valid) / len(valid)
    avg_prefill = sum(r['prefill_speed'] for r in valid) / len(valid)
    avg_decode = sum(r['decode_speed'] for r in valid) / len(valid)
    print(f"[完成] 并发 {args.concurrency} 平均结果:")
    print(f"  - 成功请求数: {len(valid)} / {args.concurrency}")
    print(f"  - 平均首包延迟 (TTFT): {avg_ttft:.3f} 秒")
    print(f"  - 平均 Prefill 速度: {avg_prefill:.2f} t/s")
    print(f"  - 平均 Token 产生速度: {avg_decode:.2f} t/s")
    return valid

def main():
    args = parse_args()
    prompt = load_prompt(args.input_file, args.loop_count)
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    all_metrics = []
    for round_idx in range(1, args.rounds + 1):
        print(f"\n[测试] 启动第 {round_idx} / {args.rounds} 轮压测...")
        if args.concurrency == 1:
            res = run_single_request(url, prompt, args.read_only, args.max_tokens)
            if res:
                all_metrics.append(res)
                print_result(res)
        else:
            all_metrics.extend(run_concurrency(url, prompt, args))
    if not all_metrics:
        sys.exit(1)

if __name__ == "__main__":
    main()
