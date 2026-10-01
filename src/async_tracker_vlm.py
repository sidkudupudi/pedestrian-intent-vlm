# yolo11_async_evaluator.py
import os
import csv
import glob
import queue
import threading
import time
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq

# --- Core Configuration Paths ---
video_directory = "./JAAD_clips"
model_dir = "./models/Qwen2-VL-2B-Merged"
output_dir = "./output_videos"
log_csv_path = "./evaluation_log.csv"

os.makedirs(output_dir, exist_ok=True)

# Thread-safe Task Queues
vlm_queue = queue.Queue(maxsize=1)
logger_queue = queue.Queue()  # Unlimited bounds for non-blocking telemetry writes
telemetry = {"latest_plan": "No pedestrians tracked yet.", "vlm_active": False}

# --- 1. Background CSV Logger Thread ---
def asynchronous_logger_worker(log_queue, csv_path):
    """Handles disk writing operations completely isolated from the video processing loop."""
    csv_headers = ["video_id", "frame_idx", "tracking_latency_ms", "vlm_triggered", "vlm_latency_ms", "generated_plan"]
    
    # Initialize/overwrite file with headers
    with open(csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow(csv_headers)
        
    while True:
        row_data = log_queue.get()
        if row_data is None:
            break  # Shutdown signal
        with open(csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(row_data)
        log_queue.task_done()

# Start Logger Thread
logger_thread = threading.Thread(target=asynchronous_logger_worker, args=(logger_queue, log_csv_path), daemon=True)
logger_thread.start()

# --- 2. Background VLM Worker Thread ---
def asynchronous_vlm_worker(in_queue, status_container):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map="auto"
        )
        # Hotfix weight tying layer mapping
        if hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings"):
            model.lm_head.weight = model.get_input_embeddings().weight
    except Exception as e:
        status_container["latest_plan"] = f"VLM Error: {e}"
        return

    prompt = "Analyze this dashcam scene. Focus on the pedestrians. Give a brief behavioral plan for the autonomous vehicle."

    while True:
        task = in_queue.get()
        if task is None:
            break
            
        frame, video_id, current_frame, track_latency = task
        status_container["vlm_active"] = True
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt}]}]
        
        start_inf = time.time()
        try:
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], images=pil_image, padding=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=40)  # Shorter token allocation caps latency
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            inf_latency = (time.time() - start_inf) * 1000
            
            status_container["latest_plan"] = output_text
            # Push payload directly to background logger queue
            logger_queue.put([video_id, current_frame, f"{track_latency:.2f}", "True", f"{inf_latency:.2f}", output_text])
            
        except Exception as e:
            status_container["latest_plan"] = f"Error: {e}"
            logger_queue.put([video_id, current_frame, f"{track_latency:.2f}", "True", "0.0", f"Error: {e}"])
            
        status_container["vlm_active"] = False
        in_queue.task_done()

# Start VLM Thread
vlm_thread = threading.Thread(target=asynchronous_vlm_worker, args=(vlm_queue, telemetry), daemon=True)
vlm_thread.start()

# --- 3. Main High-Speed Processing Pipeline ---
# Initialize YOLOv11 Small (Swapping from YOLOv8 Nano)
print("Loading YOLOv11 Small tracking weights...")
tracker_model = YOLO("yolo11s.pt") 

video_clips = sorted(glob.glob(os.path.join(video_directory, "*.mp4")))
print(f"Found {len(video_clips)} videos for evaluation sweep.")

# Select 5 random visual tracking targets
num_visualized = min(5, len(video_clips))
import random
visualize_targets = set(random.sample(video_clips, num_visualized))

for v_idx, video_file in enumerate(video_clips):
    video_id = os.path.basename(video_file)
    output_video_path = os.path.join(output_dir, f"tracked_{video_id}")
    show_window = video_file in visualize_targets
    
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    known_track_ids = set()
    frame_idx = 0
    
    vis_status = "[VISUALIZING]" if show_window else "[PROCESSING BACKGROUND]"
    print(f"[{v_idx+1}/{len(video_clips)}] {vis_status} {video_id}")
    
    if show_window:
        cv2.namedWindow(f"Live View: {video_id}", cv2.WINDOW_AUTOSIZE)
    
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        trigger_vlm = False
        
        # Track using YOLOv11 (class 0 = pedestrian)
        results = tracker_model.track(frame, persist=True, classes=[0], verbose=False)
        
        if results[0].boxes and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for tid in track_ids:
                if tid not in known_track_ids:
                    known_track_ids.add(tid)
                    trigger_vlm = True
            frame = results[0].plot()

        track_latency = (time.time() - loop_start) * 1000
        
        # Offload logic non-blockingly
        if trigger_vlm and not telemetry["vlm_active"] and not vlm_queue.full():
            vlm_queue.put((frame.copy(), video_id, frame_idx, track_latency))
        elif not trigger_vlm:
            # Push non-trigger log frame directly to background logger (takes ~0 ms)
            logger_queue.put([video_id, frame_idx, f"{track_latency:.2f}", "False", "0.0", "N/A"])

        # Display UI overlays
        status_color = (0, 0, 255) if telemetry["vlm_active"] else (0, 255, 0)
        cv2.putText(frame, f"Tracking Latency: {track_latency:.1f}ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"VLM Status: {'COMPUTING' if telemetry['vlm_active'] else 'IDLE'}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Plan: {telemetry['latest_plan'][:80]}...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        out_writer.write(frame)
        
        if show_window:
            cv2.imshow(f"Live View: {video_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out_writer.release()
                cv2.destroyAllWindows()
                exit(0)

    cap.release()
    out_writer.release()
    if show_window:
        cv2.destroyWindow(f"Live View: {video_id}")

print("\nFlushing remaining log operations...")
logger_queue.join()  # Hold until all data is written safely to file
print(f"[SUCCESS] Complete sweep saved to logs!")