#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import urllib.request

def main():
    prd_path = "/home/kunweiz/Desktop/vibe/wanxiangzhen/PRD/PRD.md"
    if not os.path.exists(prd_path):
        print(f"Error: PRD file not found at {prd_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(prd_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    print(f"Loaded PRD file, character count: {len(prompt)}")
    
    url = "http://127.0.0.1:8080/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "你是一个专业的系统架构师。请阅读下面的万象阵 PRD 文档。"},
            {"role": "user", "content": f"以下是项目文档：\n\n{prompt}\n\n请详细分析这份万象阵 PRD 的利弊，并给出深入的设计建议。"}
        ],
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream_options": {"include_usage": True}
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"}, method="POST")
    
    print("\n--- Starting Generation ---\n")
    sys.stdout.flush()
    
    start_time = time.time()
    ttft_time = None
    chunks_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    full_text = ""
    
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data:"):
                    continue
                data_json = line_str[5:].strip()
                if data_json == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_json)
                    chunks_count += 1
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")
                        if content or reasoning:
                            if ttft_time is None:
                                ttft_time = time.time()
                            token_text = reasoning if reasoning else content
                            print(token_text, end="")
                            sys.stdout.flush()
                            full_text += token_text
                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                except Exception as e:
                    pass
    except Exception as e:
        print(f"\n[Error] Request failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    end_time = time.time()
    
    total_duration = end_time - start_time
    ttft = (ttft_time - start_time) if ttft_time else total_duration
    decode_duration = total_duration - ttft
    
    if prompt_tokens == 0:
        prompt_tokens = int(len(prompt) * 0.4) # Fallback estimate
    if completion_tokens == 0:
        completion_tokens = chunks_count
        
    prefill_speed = prompt_tokens / ttft if ttft > 0 else 0
    decode_speed = completion_tokens / decode_duration if decode_duration > 0 else 0
    
    print("\n\n--- Generation Finished ---\n")
    print(f"Metrics:")
    print(f"  - Input Tokens (Prefill): {prompt_tokens}")
    print(f"  - Output Tokens (Generated): {completion_tokens}")
    print(f"  - TTFT (首包延迟): {ttft:.3f} s")
    print(f"  - Prefill Speed (PP 速度): {prefill_speed:.2f} t/s")
    print(f"  - Decode Speed (TG 速度): {decode_speed:.2f} t/s")
    print(f"  - Total Time: {total_duration:.3f} s")
    print(f"  - Final output length (chars): {len(full_text)}")
    print(f"  - Coherence Check: Output contains {full_text.count('think')} thinking tags, first 200 chars: {repr(full_text[:200])}")

if __name__ == "__main__":
    main()
