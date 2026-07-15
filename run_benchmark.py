#!/usr/bin/env python3
import json
import urllib.request
import time

# Paths
FEATURE_PATH = "/home/kunweiz/Desktop/vibe/wanxiangshu/FEATURE.md"
API_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Read FEATURE.md
print(f"Reading prompt from {FEATURE_PATH}...")
with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    prompt_content = f.read()

# Build payload
payload = {
    "messages": [
        {"role": "user", "content": prompt_content}
    ],
    "max_tokens": 2048,
    "temperature": 0.0
}

data = json.dumps(payload).encode("utf-8")

# Send request
print("Sending request to llama-server (max_tokens = 2048)...")
req = urllib.request.Request(
    API_URL,
    data=data,
    headers={"Content-Type": "application/json"}
)

start_time = time.time()
try:
    with urllib.request.urlopen(req) as response:
        res_data = response.read().decode("utf-8")
        elapsed = time.time() - start_time
        res_json = json.loads(res_data)
        
        print("\n=== Benchmark Result ===")
        print(f"Total time elapsed: {elapsed:.2f} seconds")
        
        # Extract timings and token counts
        usage = res_json.get("usage", {})
        timings = res_json.get("timings", {})
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        
        # Calculate speeds
        prompt_ms = timings.get("prompt_ms", 0.0)
        predicted_ms = timings.get("predicted_ms", 0.0)
        
        if prompt_ms > 0:
            prompt_speed = (prompt_tokens / prompt_ms) * 1000
            print(f"Prompt speed: {prompt_speed:.2f} t/s (eval time: {prompt_ms/1000:.2f}s)")
        if predicted_ms > 0:
            predicted_speed = (completion_tokens / predicted_ms) * 1000
            print(f"Generation speed: {predicted_speed:.2f} t/s (generation time: {predicted_ms/1000:.2f}s)")
            
        print(f"Overall speed (total tokens / elapsed): {total_tokens / elapsed:.2f} t/s")
        
        # Speculative stats
        draft_n = timings.get("draft_n", 0)
        draft_n_accepted = timings.get("draft_n_accepted", 0)
        if draft_n > 0:
            print(f"Speculative acceptance rate: {draft_n_accepted / draft_n * 100:.2f}% ({draft_n_accepted} / {draft_n})")
            
        # Preview response content
        choice = res_json.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        
        print("\n--- Generated Reasoning Preview (first 200 chars) ---")
        print(reasoning[:200] + "...")
        print("\n--- Generated Content Preview (first 200 chars) ---")
        print(content[:200] + "...")
        
except Exception as e:
    print(f"Error during benchmark: {e}")
