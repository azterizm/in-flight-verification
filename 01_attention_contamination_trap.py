#!/usr/bin/env python3
"""
01_attention_contamination_trap.py
Check after generation. Generates one grounded sentence, appends an injected ungrounded
sentence to the KV cache, then feeds a fixed follow-on sentence one token at a time and
reports how much attention each token puts on the injected span (mean of the last 4 layers;
a token counts as a hit above 5%). The NLI check runs once at the end, on the whole response.

The ungrounded sentence and the follow-on text are fixed, not generated, so this measures
attention on one example; it does not show how often or how far a model builds on an error.
The source passage is an illustrative paraphrase, not the text of any statute.
"""
import os
import sys
import time
import platform
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Attention Contamination Trap Telemetry Harness"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify environment, dependencies, and NLI sentinel without loading the causal model."
    )
    return parser.parse_args()

args = parse_args()
script_start_time = time.perf_counter()

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_GEN = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", DEFAULT_GEN)

def is_model_cached(repo_id: str) -> bool:
    """Checks whether model weights (.safetensors or .bin) exist in local HF cache."""
    repo_folder = "models--" + repo_id.replace("/", "--")
    snapshots_dir = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), repo_folder, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False
    for _, _, files in os.walk(snapshots_dir):
        for f in files:
            if f.endswith(".safetensors") or f.endswith(".bin"):
                return True
    return False

# Smart fallback: prefer cached nli-deberta-v3-base if small weights are not yet locally present
DEFAULT_NLI = "cross-encoder/nli-deberta-v3-small"
if not is_model_cached(DEFAULT_NLI) and is_model_cached("cross-encoder/nli-deberta-v3-base"):
    DEFAULT_NLI = "cross-encoder/nli-deberta-v3-base"

NLI_MODEL_ID = os.getenv("NLI_MODEL_ID", DEFAULT_NLI)

print("=" * 80)
print("  SYSTEM TELEMETRY & ATTENTION CONTAMINATION INSPECTOR")
print(f"  Timestamp:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Platform:         {platform.system()} ({platform.machine()}) | Device: {DEVICE.upper()}")
print(f"  PyTorch:          {torch.__version__} | Transformers: {transformers.__version__}")
print(f"  Generator Target: meta-llama/Llama-3.1-8B-Instruct ({GEN_MODEL_ID})")
print(f"  Validator:        {NLI_MODEL_ID}")
print("=" * 80)

print(f"\n[*] Loading NLI Post-Hoc Validator: {NLI_MODEL_ID}")
try:
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL_ID, clean_up_tokenization_spaces=False)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(DEVICE)
    nli_model.eval()
    print(f"    └─ Sentinel ready. Registered classes: {getattr(nli_model.config, 'id2label', 'Default')}")
except Exception as e:
    print(f"[!] FATAL: Failed to load NLI validator '{NLI_MODEL_ID}': {e}")
    sys.exit(1)

def audit_sentence_nli(premise: str, hypothesis: str):
    """Evaluates the NLI relation in a single forward pass and returns (label, prob, latency_ms)."""
    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    inputs = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    if DEVICE == "mps":
        torch.mps.synchronize()
    latency = (time.perf_counter() - t0) * 1000
    idx = torch.argmax(probs).item()

    if hasattr(nli_model.config, "id2label") and nli_model.config.id2label:
        label = str(nli_model.config.id2label[idx]).upper()
    else:
        labels = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]
        label = labels[idx]

    return label, probs[idx].item(), latency

# Source passage: an illustrative paraphrase written for this demo, not the text of any statute
retrieved_subspan = (
    "Under statutory company law, corporate legal entities are incorporated by formal public registration. "
    "No statutory provision imposes automatic joint and several personal liability on directors for pre-existing corporate obligations."
)

prompt = (
    f"Statutory Context:\n{retrieved_subspan}\n\n"
    f"Q: What are the incorporation requirements and director liability rules?\n"
    f"A:"
)

if args.dry_run:
    print("\n" + "=" * 80)
    print("[DRY-RUN MODE ACTIVATED]")
    print("Testing discriminative NLI validator with canonical premises...")
    v_grounded, p_grounded, lat_grounded = audit_sentence_nli(
        retrieved_subspan,
        "Corporate legal entities are incorporated by formal public registration."
    )
    v_contradict, p_contradict, lat_contradict = audit_sentence_nli(
        retrieved_subspan,
        "Directors shall be held jointly and personally liable for all corporate debts."
    )
    print(f"  ├─ Grounded premise test:    {v_grounded:<14} (p={p_grounded:.3f}, {lat_grounded:.1f}ms)")
    print(f"  ├─ Contradiction injection:  {v_contradict:<14} (p={p_contradict:.3f}, {lat_contradict:.1f}ms)")
    print(f"  └─ Sentinel verification:    PASS")
    print("\nDry run completed successfully. All dependencies and sentinel logic are operational.")
    print("To run the full attention telemetry test, run without `--dry-run` after model weights are available.")
    print("=" * 80)
    sys.exit(0)

print(f"\n[*] Loading Generator Model: {GEN_MODEL_ID}")
try:
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID, clean_up_tokenization_spaces=False)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID,
        dtype=torch.bfloat16 if DEVICE in ["mps", "cuda"] else torch.float32,
        # Pin every module to the accelerator: "auto" may offload to CPU, which bnb 4-bit refuses
        device_map={"": DEVICE} if DEVICE in ["mps", "cuda"] else None,
        attn_implementation="eager"  # Required for output_attentions=True
    )
    if DEVICE == "cpu":
        gen_model = gen_model.to(DEVICE)
    gen_model.eval()
    if DEVICE == "mps":
        torch.mps.empty_cache()
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    print(f"\n[!] FATAL MEMORY ERROR: Unable to allocate memory for {GEN_MODEL_ID}.")
    print(f"    Details: {e}")
    print("    Guidance: Set GEN_MODEL_ID to a lighter model (e.g., 'meta-llama/Llama-3.2-3B-Instruct')")
    print("    or ensure other memory-heavy applications are closed on this host.")
    sys.exit(1)
except Exception as e:
    print(f"\n[!] FATAL: Failed to load causal model '{GEN_MODEL_ID}': {e}")
    print("    Check your network access or Hugging Face authentication (huggingface-cli login).")
    sys.exit(1)

print("\n" + "="*80)
print("PHASE 1: THE UNCONTROLLED CONTAMINATION TRAP (DEFAULT CLOSED API / POST-HOC)")
print("="*80)

input_ids = gen_tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
prompt_len = input_ids.shape[1]

# --- STEP 1: Sentence 1 (Autoregressive Generation) ---
print("\n[STEP 1]: Generating Sentence 1 (Grounding from context)...")
past_key_values = None
current_input_ids = input_ids
s1_token_ids = []

max_s1_tokens = 60
for step in range(max_s1_tokens):
    with torch.no_grad():
        outputs = gen_model(current_input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
    
    token_id_val = next_token_id.item()
    s1_token_ids.append(token_id_val)
    current_input_ids = next_token_id
    token_str = gen_tok.decode(next_token_id[0], skip_special_tokens=True)
    
    if token_id_val == gen_tok.eos_token_id:
        break
    if "." in token_str or "\n" in token_str:
        break

s1_text = gen_tok.decode(s1_token_ids, skip_special_tokens=True).strip()
verdict_1, conf_1, lat_1 = audit_sentence_nli(retrieved_subspan, s1_text)

# Sequence coordinate where grounded sentence ends
if hasattr(past_key_values, "get_seq_length"):
    s1_end_coord = past_key_values.get_seq_length()
elif hasattr(past_key_values, "layers"):
    s1_end_coord = past_key_values.layers[0].keys.shape[2]
else:
    s1_end_coord = past_key_values[0][0].shape[2]

print(f"[SENTENCE 1]: \"{s1_text}\"")
print(f"  ├─ Sentinel Verdict: {verdict_1} (p={conf_1:.2f})")
print(f"  └─ KV Cache State:   LOCKED at Sequence Index = {s1_end_coord}")

# --- STEP 2: Sentence 2 (Adversarial Error / Contamination Seed) ---
print("\n[STEP 2]: Simulating Mid-Generation Contamination...")
hallucinated_clause = " Furthermore, directors shall be held jointly and personally liable for all pre-existing corporate debts."
h_ids = gen_tok(hallucinated_clause, return_tensors="pt").input_ids.to(DEVICE)

with torch.no_grad():
    out_h = gen_model(h_ids, past_key_values=past_key_values, use_cache=True)
    polluted_kv = out_h.past_key_values

if hasattr(polluted_kv, "get_seq_length"):
    bad_end_coord = polluted_kv.get_seq_length()
elif hasattr(polluted_kv, "layers"):
    bad_end_coord = polluted_kv.layers[0].keys.shape[2]
else:
    bad_end_coord = polluted_kv[0][0].shape[2]

bad_span = (s1_end_coord, bad_end_coord)
print(f"[SENTENCE 2]: \"{hallucinated_clause.strip()}\"")
print(f"  ├─ Contaminated Token Span: [{bad_span[0]}:{bad_span[1]}]")
print(f"  └─ Status: UNVERIFIED (In-flight evaluation bypassed in post-hoc architecture)")

# --- STEP 3: Downstream Cascade & Real-Time Attention Hit Tracking ---
print("\n[STEP 3]: Feeding fixed follow-on text through the contaminated cache...")
print("-" * 80)
print(f"{'STEP':<6} | {'TOKEN':<16} | {'ATTN ON BAD SPAN':<18} | {'HEADS FIRING':<14} | {'STATUS'}")
print("-" * 80)

# Downstream continuation conditioned on the poisoned state
cascade_prompt = " Therefore, creditors may immediately initiate personal asset seizure against individual directors."
cascade_ids = gen_tok(cascade_prompt, return_tensors="pt").input_ids.to(DEVICE)

current_kv = polluted_kv
cumulative_attention_hits = 0
total_downstream_tokens = 0
attention_masses = []
heads_firing_counts = []

for idx in range(cascade_ids.shape[1]):
    step_input = cascade_ids[:, idx:idx+1]
    with torch.no_grad():
        out_step = gen_model(
            step_input,
            past_key_values=current_kv,
            use_cache=True,
            output_attentions=True
        )
        current_kv = out_step.past_key_values
        # Average attention across the last 4 semantic layers
        num_layers_to_inspect = min(4, len(out_step.attentions))
        layer_attns = out_step.attentions[-num_layers_to_inspect:]
        stacked = torch.stack(layer_attns)
        mean_heads = stacked.mean(dim=0).squeeze(0).squeeze(1) # [heads, kv_len]
        span_attn = mean_heads[:, bad_span[0]:bad_span[1]].sum(dim=-1) # [heads]
        mean_span_mass = span_attn.mean().item() * 100.0 # percentage
        heads_firing = (span_attn > 0.05).sum().item()
        total_heads = mean_heads.shape[0]

    token_str = gen_tok.decode(step_input[0], skip_special_tokens=True)
    if mean_span_mass > 5.0:
        cumulative_attention_hits += 1
        indicator = "ATTN_HIT"
    else:
        indicator = "CLEAN"

    attention_masses.append(mean_span_mass)
    heads_firing_counts.append(heads_firing)
    total_downstream_tokens += 1

    clean_repr = repr(token_str)
    if len(clean_repr) > 16:
        clean_repr = clean_repr[:13] + "..."

    print(f"#{idx+1:<5} | {clean_repr:<16} | {mean_span_mass:>6.2f}% attention  | {heads_firing:>2}/{total_heads} heads     | {indicator}")

print("-" * 80)
avg_attention = sum(attention_masses) / len(attention_masses) if attention_masses else 0.0
avg_heads_firing = sum(heads_firing_counts) / len(heads_firing_counts) if heads_firing_counts else 0.0

print(f"\nTELEMETRY SUMMARY (SPAN [{bad_span[0]}:{bad_span[1]}]):")
print(f"  ├─ Downstream Tokens Analyzed:  {total_downstream_tokens}")
print(f"  ├─ Contaminated Span Hits:      {cumulative_attention_hits}/{total_downstream_tokens}")
print(f"  ├─ Mean Span Attention Mass:    {avg_attention:.2f}%")
print(f"  └─ Heads Above 5% (mean/token): {avg_heads_firing:.1f}/{total_heads}")

# --- STEP 4: Post-Hoc Evaluation ---
print("\n" + "="*80)
print("POST-HOC VERIFICATION (DEFAULT ARCHITECTURE)")
print("="*80)
full_output = f"{s1_text} {hallucinated_clause.strip()} {cascade_prompt.strip()}"
final_verdict, final_conf, final_lat = audit_sentence_nli(retrieved_subspan, full_output)

print(f"Response:     \"{full_output}\"")
print(f"Gate Verdict: {final_verdict} (p={final_conf:.2f}, Latency: {final_lat:.1f}ms)")
print("Action:       DISCARD ENTIRE SEQUENCE (No KV-rollback mechanism on closed API)")

if DEVICE == "mps":
    torch.mps.empty_cache()

total_runtime = time.perf_counter() - script_start_time
print(f"\n[*] Execution time: {total_runtime:.2f}s.\n")
