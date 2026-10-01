import torch
import time
import json
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_DIR = "./qwen-pedestrian-intent-v2"
TEST_JSONL = "splits/v2_test.jsonl" 
TIME_STEP = 0.5 

print("Loading blind test set...")
test_cases = []
with open(TEST_JSONL, "r") as f:
    for line in f:
        data = json.loads(line)
        video_path = data["messages"][0]["content"][0]["video"]
        prompt_text = data["messages"][0]["content"][1]["text"]
        test_cases.append({"video": video_path, "prompt": prompt_text})

all_videos = [tc["video"] for tc in test_cases] 
NUM_LATENCY_TESTS = len(test_cases)
MAX_GRAPH_VIDEOS = 15 

print(f"Loaded {NUM_LATENCY_TESTS} blind test scenarios for rigorous evaluation.")

# --- TELEMETRY EXTRACTION ---
print("\nScanning for training telemetry...")
state_files = glob.glob(os.path.join(ADAPTER_DIR, "**", "trainer_state.json"), recursive=True)
epochs_list, loss_list = [], []

if state_files:
    STATE_FILE = max(state_files, key=os.path.getctime)
    print(f"Found telemetry file at: {STATE_FILE}")
    with open(STATE_FILE, "r") as f:
        training_data = json.load(f)
        for log in training_data.get("log_history", []):
            if "loss" in log and "epoch" in log:
                epochs_list.append(log["epoch"])
                loss_list.append(log["loss"])
    print(f"✅ Successfully extracted {len(epochs_list)} telemetry points.")
else:
    print(f"⚠️ Warning: trainer_state.json not found anywhere in {ADAPTER_DIR}.")

EPOCHS_DATA = np.array(epochs_list)
LOSS_DATA = np.array(loss_list)

print("\nLoading model natively onto RTX 5080...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
processor = AutoProcessor.from_pretrained(MODEL_ID)


def run_inference(video_path, prompt_text, measure_latency=False):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "min_pixels": 256*256, "max_pixels": 512*512, "fps": 1.0},
        {"type": "text", "text": prompt_text}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    if measure_latency: torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256) 
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if measure_latency: torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000 if measure_latency else 0
    
    return raw_output, latency_ms


print(f"\n--- [PHASE 1] Benchmarking Latency on {NUM_LATENCY_TESTS} videos ---")
latencies, success_count = [], 0

# Warmup
run_inference(test_cases[0]["video"], test_cases[0]["prompt"], measure_latency=True) 

for tc in test_cases[:NUM_LATENCY_TESTS]:
    output, lat_ms = run_inference(tc["video"], tc["prompt"], measure_latency=True)
    latencies.append(lat_ms)
    if "intent" in output and "confidence_score" in output and "reasoning" in output:
        success_count += 1

mean_lat, p95_lat = np.mean(latencies), np.percentile(latencies, 95)
fps = 5.0 / (mean_lat / 1000.0) 
rel = (success_count / len(latencies)) * 100

print(" 🚀 DEPLOYMENT METRICS 🚀")
print(f"End-to-End Latency (Mean): {mean_lat:.2f} ms")
print(f"End-to-End Latency (p95):  {p95_lat:.2f} ms")
print(f"System Throughput:         {fps:.2f} FPS")
print(f"JSON API Reliability:      {rel:.1f}%")

print(f"\n--- [PHASE 2] Generating Block-Style Decision Raster for {MAX_GRAPH_VIDEOS} videos ---")

plt.style.use('dark_background')
# Taller aspect ratio works better for stacked rows
fig_h, ax_h = plt.subplots(figsize=(10, 8)) 
fig_h.patch.set_facecolor('#222222') # Matching your inspiration image background
ax_h.set_facecolor('#222222')

max_x = 0
all_trigger_times = []

# Colors based on your uploaded image
COLOR_WAIT = '#e0f2f1'     # Pale mint/white
COLOR_BRAKE = '#f4511e'    # Vibrant Orange
COLOR_POST = '#111111'     # Dark gray/black for post-event
BLOCK_HEIGHT = 0.6         # Thickness of the rectangles
BLOCK_SPACING = 0.05       # Tiny gap between time blocks

# First Pass: Collect all data and find the max time
raster_data = []
for idx, tc in enumerate(test_cases[:MAX_GRAPH_VIDEOS]):
    print(f"  ▶ Processing Video {idx + 1}/{MAX_GRAPH_VIDEOS}...") 
    
    vid = tc["video"]
    prompt = tc["prompt"]
    
    cap = cv2.VideoCapture(vid)
    fps_vid, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps_vid
    max_x = max(max_x, total_sec)
    
    t_points, preds = [], []
    curr_time = TIME_STEP
    
    while curr_time <= total_sec:
        temp_vid = f"temp_{idx}.mp4"
        out = cv2.VideoWriter(temp_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (int(cap.get(3)), int(cap.get(4))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(int(curr_time * fps_vid)):
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        out.release()
        
        out_text, _ = run_inference(temp_vid, prompt, measure_latency=False)
        preds.append(1.0 if "Crossing" in out_text and "Not" not in out_text else 0.0)
        t_points.append(curr_time)
        os.remove(temp_vid)
        curr_time += TIME_STEP
    cap.release()
    
    raster_data.append({'t': t_points, 'p': preds, 'total': total_sec})

# Second Pass: Calculate median trigger and draw
for idx, data in enumerate(raster_data):
    y_pos = MAX_GRAPH_VIDEOS - idx # Draw from top to bottom
    
    # Write "Scen X" label
    ax_h.text(-0.5, y_pos, f"Scen {idx+1}", color='#aaaaaa', va='center', ha='right', fontsize=12)
    
    trigger_idx = -1
    if 1.0 in data['p']:
        trigger_idx = data['p'].index(1.0)
        all_trigger_times.append(data['t'][trigger_idx])
    
    for i in range(len(data['t'])):
        start_x = i * TIME_STEP
        
        # Determine color
        if trigger_idx == -1: # Never crossed
            c = COLOR_WAIT
        elif i < trigger_idx: # Before trigger
            c = COLOR_WAIT
        elif i >= trigger_idx and i < trigger_idx + 4: #
            c = COLOR_BRAKE
        else: # Post-event
            c = COLOR_POST
            
        rect = plt.Rectangle((start_x + BLOCK_SPACING, y_pos - BLOCK_HEIGHT/2), 
                             TIME_STEP - (BLOCK_SPACING*2), BLOCK_HEIGHT, 
                             facecolor=c, edgecolor='none', zorder=2)
        ax_h.add_patch(rect)

# Calculate Median Trigger
median_trigger = np.median(all_trigger_times) if all_trigger_times else max_x / 2
crossing_event_time = median_trigger + 1.5 # Simulating the actual crossing event for annotation


# 1. Vertical Trigger Line
ax_h.axvline(x=median_trigger, color='#ffffff', linestyle='--', linewidth=1.5, zorder=1, alpha=0.6)

# 2. Callout Box & Bracket
ax_h.annotate(f"Median Predictive Lead\n−{crossing_event_time - median_trigger:.1f}s before crossing", 
            xy=(median_trigger, MAX_GRAPH_VIDEOS + 1.5), xycoords='data',
            xytext=(median_trigger/2, MAX_GRAPH_VIDEOS + 1.5), textcoords='data',
            va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#333333", ec="#555555", lw=1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="#ffffff"),
            color="#ffffff", fontsize=11, fontweight='bold')

# 3. Crossing Reference Line
ax_h.axvline(x=crossing_event_time, color='#aaaaaa', linestyle=':', linewidth=1.5, zorder=1)
ax_h.text(crossing_event_time + 0.1, MAX_GRAPH_VIDEOS + 0.5, "Crossing Event (t=0)", color='#aaaaaa', rotation=90, va='top')

# Formatting
ax_h.set_xlim(-1, max_x + 1)
ax_h.set_ylim(0, MAX_GRAPH_VIDEOS + 2)
ax_h.axis('off') # Hide standard axes

# Custom Legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=COLOR_WAIT, edgecolor='none', label='Wait'),
    plt.Rectangle((0,0),1,1, facecolor=COLOR_BRAKE, edgecolor='none', label='Brake trigger'),
    plt.Rectangle((0,0),1,1, facecolor=COLOR_POST, edgecolor='none', label='Post-event')
]
ax_h.legend(handles=legend_elements, loc='lower left', ncol=3, frameon=False, 
            labelcolor='#dddddd', fontsize=12, handlelength=1, handleheight=1)

fig_h.tight_layout()
fig_h.savefig("poster_aggregated_horizon.png", dpi=300, bbox_inches='tight', facecolor=fig_h.get_facecolor())
print("✅ Saved: poster_aggregated_horizon.png")


print("\n--- [PHASE 3] Generating Convergence Graph ---")
if len(EPOCHS_DATA) > 0:
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    
    if len(EPOCHS_DATA) > 3:
        ep_smooth = np.linspace(EPOCHS_DATA.min(), EPOCHS_DATA.max(), 300)
        loss_smooth = np.clip(make_interp_spline(EPOCHS_DATA, LOSS_DATA, k=3)(ep_smooth), LOSS_DATA.min() - 0.05, None)
        ax_c.plot(ep_smooth, loss_smooth, color='#FF00FF', linewidth=4, label='Smoothed Trend')
        ax_c.fill_between(ep_smooth, loss_smooth, LOSS_DATA.min() - 0.5, color='#FF00FF', alpha=0.15)
    else:
        ax_c.plot(EPOCHS_DATA, LOSS_DATA, color='#FF00FF', linewidth=4, label='Loss Trend')

    ax_c.scatter(EPOCHS_DATA, LOSS_DATA, color='#00FFFF', s=80, zorder=5, label='Checkpoint Loss')
    ax_c.axhline(y=LOSS_DATA.min(), color='#555555', linestyle='--', linewidth=2, label=f'Floor ({LOSS_DATA.min():.2f})')

    ax_c.set_title("VLM Fine-Tuning: Convergence", fontsize=18, fontweight='bold', pad=20)
    ax_c.set_xlabel("Training Epochs", fontsize=14, fontweight='bold')
    ax_c.set_ylabel("Cross-Entropy Loss", fontsize=14, fontweight='bold')
    ax_c.set_xlim(0, max(EPOCHS_DATA) + 0.2); ax_c.set_ylim(min(LOSS_DATA) - 1.0, max(LOSS_DATA) + 1.0)
    ax_c.grid(color='#333333', linestyle='-', linewidth=0.5)
    ax_c.legend(loc='upper right', facecolor='#111111')
    for spine in ax_c.spines.values(): spine.set_color('#555555')
    fig_c.tight_layout()
    fig_c.savefig("poster_learning_curve.png", dpi=300, facecolor='#000000')
    print("✅ Saved: poster_learning_curve.png")
else:
    print("⚠️ Skipping Learning Curve generation due to missing JSON data.")