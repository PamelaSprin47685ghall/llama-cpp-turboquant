#!/usr/bin/env python3
import json
import urllib.request

FEATURE_PATH = "/home/kunweiz/Desktop/vibe/wanxiangshu/FEATURE.md"
API_URL = "http://127.0.0.1:8080/v1/chat/completions"

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    prompt_content = f.read()

payload = {
    "messages": [
        {"role": "user", "content": prompt_content}
    ],
    "max_tokens": 800,
    "temperature": 0.0
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    API_URL,
    data=data,
    headers={"Content-Type": "application/json"}
)

try:
    print("Requesting 800 tokens to check for gibberish...")
    with urllib.request.urlopen(req) as response:
        res_json = json.loads(response.read().decode("utf-8"))
        choice = res_json.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        timings = res_json.get("timings", {})
        
        print("\n=== TIMINGS ===")
        print(f"Draft generated: {timings.get('draft_n')}, accepted: {timings.get('draft_n_accepted')}")
        if timings.get('draft_n', 0) > 0:
            print(f"Acceptance rate: {timings.get('draft_n_accepted')/timings.get('draft_n')*100:.2f}%")
            
        print("\n=== REASONING CONTENT ===")
        print(reasoning[:1500])
        if len(reasoning) > 1500:
            print("... [TRUNCATED REASONING] ...")
            
        print("\n=== GENERATED CONTENT ===")
        print(content[:1500])
        if len(content) > 1500:
            print("... [TRUNCATED CONTENT] ...")
            
        # Check if content has gibberish character patterns or is empty
        if not content:
            print("\nWARNING: Content is completely empty!")
        else:
            print(f"\nContent length: {len(content)} characters.")
            
except Exception as e:
    print(f"Error: {e}")
