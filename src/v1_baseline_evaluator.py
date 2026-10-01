import torch
import time
import json
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
TEST_JSONL = "splits/v2_test.jsonl" 

print("🚀 Initializing V1 Baseline Evaluator (Reactive Architecture)...")

# Load only the video paths from the blind test set
test_videos = []
with open(TEST_JSONL, "r") as f:
    for line in f:
        data = json.loads(line)
        video_path = data["messages"][0]["content"][0]["video"]
        test_videos.append(video_path)

print(f"Loaded {len(test_videos)} blind test scenarios.")

# ==========================================
# 2. LOAD BASE MODEL (NO ADAPTERS)
# ==========================================
print("\nLoading Base Model natively onto RTX 5080...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
# Notice we are NOT wrapping this in PeftModel
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ==========================================
# 3. V1 INFERENCE ENGINE (BINARY PROMPT)
# ==========================================
def run_v1_inference(video_path):
    # The V1 Prompt: No JSON, no reasoning, just a binary classification
    v1_prompt = "You are an autonomous driving system. Will the pedestrian in this video cross the road? Answer ONLY with 'Crossing' or 'Not Crossing'."
    
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "min_pixels": 256*256, "max_pixels": 512*512, "fps": 1.0},
        {"type": "text", "text": v1_prompt}
    ]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # Drastically reduced max_tokens because we only expect 1-2 words
        generated_ids = model.generate(**inputs, max_new_tokens=10) 
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return raw_output, latency_ms

# ==========================================
# 4. EXECUTE BENCHMARK
# ==========================================
print("\n--- Benchmarking V1 Latency ---")
latencies = []

# Warmup
run_v1_inference(test_videos[0])

for vid in tqdm(test_videos, desc="Running V1 Baseline"):
    output, lat_ms = run_v1_inference(vid)
    latencies.append(lat_ms)

mean_lat = np.mean(latencies)
p95_lat = np.percentile(latencies, 95)
fps = 5.0 / (mean_lat / 1000.0) # Assuming 5 frames per context window

print("\n 🛑 V1 BASELINE METRICS (RTX 5080) 🛑")
print(f"End-to-End Latency (Mean): {mean_lat:.2f} ms")
print(f"End-to-End Latency (p95):  {p95_lat:.2f} ms")
print(f"System Throughput:         {fps:.2f} FPS")
print("\nNow you have the exact numbers to complete your poster's comparison table!")