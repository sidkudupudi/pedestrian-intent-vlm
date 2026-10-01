# calculate_benchmarks.py
import pandas as pd
import numpy as np

log_csv_path = "./evaluation_log.csv"

try:
    df = pd.read_csv(log_csv_path)
except FileNotFoundError:
    print(f"[ERROR] Could not find {log_csv_path}. Please check file location.")
    exit(1)

# 1. Total Scoped Dataset Metadata
total_frames = len(df)
unique_videos = df["video_id"].nunique()
print(f"Total Video Clips Processed : {unique_videos}")
print(f"Total Video Frames Evaluated: {total_frames}")

print("\n--- LATENCY PERFORMANCES ---")
# 2. Tracking Loop Performance (Target: <5ms for high frequency vehicle loops)
mean_track_lat = df["tracking_latency_ms"].mean()
p95_track_lat = np.percentile(df["tracking_latency_ms"], 95)
effective_fps = 1000.0 / mean_track_lat

print(f"Mean Tracker Latency        : {mean_track_lat:.2f} ms ({effective_fps:.1f} Hz Control Loop)")
print(f"95th Percentile Latency     : {p95_track_lat:.2f} ms")

# 3. VLM Asynchronous Layer Performance
vlm_triggered_df = df[df["vlm_triggered"] == True]
total_triggers = len(vlm_triggered_df)

if total_triggers > 0:
    mean_vlm_lat = vlm_triggered_df["vlm_latency_ms"].mean()
    p95_vlm_lat = np.percentile(vlm_triggered_df["vlm_latency_ms"], 95)
    print(f"Mean VLM Intent Inference   : {mean_vlm_lat:.2f} ms")
    print(f"95th Percentile VLM Latency : {p95_vlm_lat:.2f} ms")
else:
    print("VLM Latency                 : N/A (No trigger events logged)")

print("\n--- SYSTEM DUTY CYCLE & EFFICIENCY ---")
trigger_rate = (total_triggers / total_frames) * 100
compute_reduction = 100.0 - trigger_rate

print(f"Total VLM Trigger Events    : {total_triggers} times across dataset")
print(f"VLM Activation Duty Cycle   : {trigger_rate:.2f}% of total timeline")
print(f"Compute Resource Savings    : {compute_reduction:.2f}% compared to frame-by-frame VLM deployment")
