"""
Held-out evaluation of pedestrian crossing-intent prediction on v2_test.jsonl (131 clips never used for LoRA training).

Compares the LoRA fine-tuned Qwen2-VL-2B (V2) against the same base model zero-shot, using the exact
prompt, video sampling (fps=1, 256^2-512^2 pixels) and 4-bit NF4 loading used by master_evaluator.py.
Greedy decoding (do_sample=False) for reproducibility.

Usage (from the repository root, with v2_training_clips/ present):
    python src/eval_intent_heldout.py --mode lora   [--limit N]
    python src/eval_intent_heldout.py --mode base   [--limit N]
"""
import argparse, json, re, time, os
import numpy as np, pandas as pd, torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_DIR = "./qwen-pedestrian-intent-v2"
TEST_JSONL = "splits/v2_test.jsonl"
OUT_DIR = "results/metrics"


def parse_intent(text):
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(clean)
        intent = str(obj.get("intent", ""))
        if intent in ("Crossing", "Not Crossing"):
            return intent, True
    except Exception:
        pass
    # Fallback for non-JSON answers: look for the label words
    if re.search(r"not\s+crossing", clean, re.I):
        return "Not Crossing", False
    if re.search(r"crossing", clean, re.I):
        return "Crossing", False
    return "Unparsed", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lora", "base"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train-preproc", action="store_true",
                    help="encode clips exactly as during training (processor defaults: fps 2, default resolution) "
                         "instead of master_evaluator's fps 1 / 256^2-512^2 pixels")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(TEST_JSONL)]
    if args.limit:
        cases = cases[: args.limit]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    if args.mode == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    rows = []
    for i, ex in enumerate(cases):
        video = ex["messages"][0]["content"][0]["video"]
        prompt = ex["messages"][0]["content"][1]["text"]
        truth = json.loads(ex["messages"][1]["content"][0]["text"])["intent"]
        vid = {"type": "video", "video": video} if args.train_preproc else \
              {"type": "video", "video": video, "min_pixels": 256 * 256, "max_pixels": 512 * 512, "fps": 1.0}
        messages = [{"role": "user", "content": [vid, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 1000
        out = processor.batch_decode([g[len(x):] for x, g in zip(inputs.input_ids, gen)], skip_special_tokens=True)[0]
        pred, valid_json = parse_intent(out)
        rows.append(dict(clip=os.path.basename(video), truth=truth, pred=pred, correct=pred == truth,
                         valid_json=valid_json, latency_ms=round(lat, 1), raw_output=out.strip()))
        print(f"[{args.mode}] {i+1}/{len(cases)} truth={truth:12s} pred={pred:12s} json={valid_json} {lat:.0f} ms", flush=True)

    df = pd.DataFrame(rows)
    tag = args.mode + ("_trainpreproc" if args.train_preproc else "")
    df.to_csv(f"{OUT_DIR}/heldout_predictions_{tag}.csv", index=False)

    y, p = df.truth.values, df.pred.values
    recalls = {c: float(((p == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in ("Crossing", "Not Crossing")}
    f1s = {}
    for c in ("Crossing", "Not Crossing"):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f1s[c] = float(2 * tp / max(2 * tp + fp + fn, 1))
    summary = dict(
        mode=tag, n=len(df), accuracy=float(df.correct.mean()),
        balanced_accuracy=float(np.mean(list(recalls.values()))), macro_f1=float(np.mean(list(f1s.values()))),
        recall_crossing=recalls["Crossing"], recall_not_crossing=recalls["Not Crossing"],
        valid_json_rate=float(df.valid_json.mean()), unparsed=int((df.pred == "Unparsed").sum()),
        majority_baseline_accuracy=float((y == pd.Series(y).mode()[0]).mean()),
        latency_mean_ms=float(df.latency_ms.mean()), latency_p95_ms=float(np.percentile(df.latency_ms, 95)),
        confusion={f"{t}->{q}": int(((y == t) & (p == q)).sum()) for t in ("Crossing", "Not Crossing") for q in ("Crossing", "Not Crossing", "Unparsed")},
    )
    json.dump(summary, open(f"{OUT_DIR}/heldout_summary_{tag}.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
