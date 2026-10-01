import torch
import json
import cv2
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_DIR = "./qwen-pedestrian-intent-v2"  
VIDEO_DIR = "v2_training_clips/*.mp4"
OUTPUT_DIR = "./v2_dashboard_visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading V2 Model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
processor = AutoProcessor.from_pretrained(MODEL_ID)

video_files = glob.glob(VIDEO_DIR)
print(f"Found {len(video_files)} videos. Starting batch generation...")

for video_path in tqdm(video_files, desc="Rendering Dashboards"):
    video_filename = os.path.basename(video_path).replace('.mp4', '')
    
    prompt_text = (
        "You are an autonomous driving system. The ego-vehicle is moving fast. "
        "The traffic light is none. Analyze the pedestrian in this video. "
        "Respond ONLY with a valid JSON object matching the exact schema: "
        "{\"pedestrian_demographics\": \"<text>\", \"pedestrian_appearance\": \"<text>\", \"pedestrian_attention\": \"<text>\", "
        "\"pedestrian_motion\": \"<text>\", \"intent\": \"Crossing\" | \"Not Crossing\", \"confidence_score\": <float>, \"reasoning\": \"<text>\"}."
    )

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "min_pixels": 256*256, "max_pixels": 512*512, "fps": 1.0},
        {"type": "text", "text": prompt_text}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    clean_json = raw_output.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_json)
    except Exception as e:
        data = {"error": "Invalid Output", "raw_text": clean_json}

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        continue 

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.style.use('dark_background')
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.5, 1]})
    fig.patch.set_facecolor('#0a0a0a')

    ax_img.imshow(frame_rgb)
    ax_img.axis('off')
    ax_img.set_title("Vehicle Camera Feed", fontsize=16, fontweight='bold', color='#ffffff', pad=15)

    ax_text.axis('off')
    ax_text.set_title("V2 Perception Logic (Chain-of-Thought)", fontsize=16, fontweight='bold', color='#00ffff', pad=15)

    y_offset = 0.95
    line_height = 0.08

    def add_stat(label, value, color="#ffffff", highlight=False):
        global y_offset
        weight = 'bold' if highlight else 'normal'
        val_color = '#00ffff' if highlight else color
        ax_text.text(0.05, y_offset, f"{label}:", fontsize=14, fontweight='bold', color='#888888', transform=ax_text.transAxes)
        ax_text.text(0.35, y_offset, str(value), fontsize=14, fontweight=weight, color=val_color, transform=ax_text.transAxes, wrap=True)
        y_offset -= line_height

    add_stat("Demographics", data.get("pedestrian_demographics", "N/A"))
    add_stat("Appearance", data.get("pedestrian_appearance", "N/A"))
    add_stat("Attention", data.get("pedestrian_attention", "N/A"), highlight=(data.get("pedestrian_attention") == "Looking at vehicle"))
    add_stat("Motion", data.get("pedestrian_motion", "N/A"))
    y_offset -= 0.05 

    intent_val = data.get("intent", "N/A")
    intent_color = '#ff0055' if intent_val == "Crossing" else '#00ffaa'
    add_stat("INTENT", intent_val, color=intent_color, highlight=True)
    add_stat("CONFIDENCE", f"{data.get('confidence_score', 0.0) * 100:.1f}%" if isinstance(data.get('confidence_score'), (int, float)) else "N/A")
    y_offset -= 0.05 

    ax_text.text(0.05, y_offset, "CAUSAL REASONING:", fontsize=12, fontweight='bold', color='#888888', transform=ax_text.transAxes)
    ax_text.text(0.05, y_offset - 0.05, data.get("reasoning", "N/A"), fontsize=12, color='#cccccc', transform=ax_text.transAxes, wrap=True, ha='left', va='top', bbox=dict(facecolor='#1a1a1a', edgecolor='#333333', boxstyle='round,pad=1'))

    plt.tight_layout()
    output_filepath = os.path.join(OUTPUT_DIR, f"dashboard_{video_filename}.png")
    fig.savefig(output_filepath, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

