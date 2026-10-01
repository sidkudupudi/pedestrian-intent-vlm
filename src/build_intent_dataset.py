import os
import cv2
import json
import random
from tqdm import tqdm
from jaad_data import JAAD

# --- CONFIGURATION ---
JAAD_DATA_PATH = "." # Root folder containing JAAD_clips and the XML annotation folders
OUTPUT_CLIPS_DIR = "./v2_training_clips"
OUTPUT_JSONL = "qwen_v2_training_data.jsonl"
FRAMES_PER_CLIP = 15 # 1.5 seconds at 10 FPS

os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)

print("Parsing all XML annotation folders and merging database...")
dataset = JAAD(data_path=JAAD_DATA_PATH)
db = dataset.generate_database()

training_examples = []

print(f"\nExtracting hyper-contextual training clips...")
for vid_id, vid_data in tqdm(db.items()):
    raw_video_path = os.path.join(JAAD_DATA_PATH, "JAAD_clips", f"{vid_id}.mp4")
    if not os.path.exists(raw_video_path):
        continue
        
    cap = cv2.VideoCapture(raw_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 1. Frame-Level Environmental Data
    traffic_annotations = vid_data.get('traffic_annotations', {})
    vehicle_annotations = vid_data.get('vehicle_annotations', {})
    
    for ped_id, ped_data in vid_data['ped_annotations'].items():
        if 'b' not in ped_id: continue # Only train on interactive pedestrians
            
        frames = ped_data.get('frames', [])
        if len(frames) < FRAMES_PER_CLIP: continue
            
        attributes = ped_data.get('attributes', {})
        appearance = ped_data.get('appearance', {})
        
        decision_frame = attributes.get('decision_point', -1)
        if decision_frame == -1: decision_frame = frames[-1]
            
        try: d_idx = frames.index(decision_frame)
        except ValueError: continue
            
        if d_idx < FRAMES_PER_CLIP: continue
            
        start_idx = d_idx - FRAMES_PER_CLIP
        clip_frames = frames[start_idx:d_idx]
        
        # --- BUILDING THE CONTEXT PROMPT ---
        
        # Ego-Vehicle
        ego_state_code = vehicle_annotations.get(decision_frame, 0)
        ego_map = {0: 'stopped', 1: 'moving slowly', 2: 'moving fast', 3: 'decelerating', 4: 'accelerating'}
        ego_state = ego_map.get(ego_state_code, 'moving')
        
        # Traffic & Infrastructure
        frame_traffic = traffic_annotations.get(decision_frame, {})
        traffic_code = frame_traffic.get('traffic_light', 0)
        light_map = {0: 'none', 1: 'red', 2: 'green'}
        traffic_light = light_map.get(traffic_code, 'none')
        has_stop_sign = frame_traffic.get('stop_sign', 0) == 1
        has_crosswalk = frame_traffic.get('ped_crossing', 0) == 1
        
        # Infrastructure narrative
        infra_text = f"The traffic light is {traffic_light}."
        if has_stop_sign: infra_text += " There is a stop sign."
        if has_crosswalk: infra_text += " There is a designated crosswalk."

        # Demographics & Appearance
        age_map = {0: 'child', 1: 'young adult', 2: 'adult', 3: 'senior'}
        gender_map = {0: 'person', 1: 'female', 2: 'male'}
        demo_text = f"{age_map.get(attributes.get('age', 2), 'adult')} {gender_map.get(attributes.get('gender', 0), 'person')}"
        
        # Parse complex appearance arrays (using the state at the decision frame)
        app_text = ""
        if appearance and len(appearance.get('pose_front', [])) > d_idx:
            if appearance['backpack'][d_idx] == 1: app_text += "wearing a backpack, "
            if appearance['phone'][d_idx] == 1: app_text += "looking at a phone, "
            if appearance['pose_front'][d_idx] == 1: app_text += "facing the camera, "
        if not app_text: app_text = "standard clothing, "
        
        # Motion & Attention
        action_code = ped_data.get('behavior', {}).get('action', [0])[d_idx]
        motion_state = "Walking" if action_code == 1 else "Standing"
        look_code = ped_data.get('behavior', {}).get('look', [0])[d_idx]
        attention_state = "Looking at vehicle" if look_code == 1 else "Not looking at vehicle"
        
        # Intent & Target JSON
        is_crossing = attributes.get('crossing', 0) > 0
        intent_label = "Crossing" if is_crossing else "Not Crossing"
        confidence = round(random.uniform(0.85, 0.99), 2) 
        
        target_json = {
            "pedestrian_demographics": demo_text.title(),
            "pedestrian_appearance": app_text.strip(", "),
            "pedestrian_attention": attention_state,
            "pedestrian_motion": motion_state,
            "intent": intent_label,
            "confidence_score": confidence,
            "reasoning": f"The {demo_text} is {motion_state.lower()} and {attention_state.lower()}. The ego-vehicle is {ego_state}."
        }
        
        # The Hyper-Contextual Prompt
        prompt_text = (
            f"You are an autonomous driving system. The ego-vehicle is {ego_state}. "
            f"{infra_text} Analyze the pedestrian in this video. "
            "Respond ONLY with a valid JSON object matching the exact schema: "
            "{\"pedestrian_demographics\": \"<text>\", \"pedestrian_appearance\": \"<text>\", \"pedestrian_attention\": \"<text>\", "
            "\"pedestrian_motion\": \"<text>\", \"intent\": \"Crossing\" | \"Not Crossing\", \"confidence_score\": <float>, \"reasoning\": \"<text>\"}."
        )
        
        # --- WRITE THE CLIP ---
        clip_filename = f"{vid_id}_{ped_id}.mp4"
        clip_path = os.path.join(OUTPUT_CLIPS_DIR, clip_filename)
        
        if not os.path.exists(clip_path):
            out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_frames[0])
            for _ in range(FRAMES_PER_CLIP):
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
            out.release()
            
        conversation = {
            "messages": [
                {"role": "user", "content": [{"type": "video", "video": clip_path}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target_json)}]}
            ]
        }
        training_examples.append(conversation)
        
    cap.release()

print(f"\nWriting {len(training_examples)} hyper-contextual V2 examples to {OUTPUT_JSONL}...")
with open(OUTPUT_JSONL, "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

print("✅ V2 Dataset Generation Complete!")