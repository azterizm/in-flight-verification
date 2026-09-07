# In-Flight Verification Control Specification for Legal RAG

This repository contains the empirical proof of mechanism and reproducible test harnesses supporting the case study:
**"Generation-Tier Integrity in Legal RAG: An In-Flight Verification Control Specification."**

---

## 1. The Core Finding & The Problem

Commercial legal RAG platforms built on closed frontier APIs (OpenAI / Anthropic) default to a **"generate-then-validate"** pattern. 

In autoregressive generation, causal attention is mathematically irreversible. When a model produces a hallucinated statutory holding mid-generation (occurring on 17–33% of specialized legal queries per Stanford RegLab benchmarks), those contaminated tokens are committed directly into GPU memory. Downstream attention matrices condition directly on that fabrication—deriving downstream legal remedies and non-existent precedent to rationalize the seed error (the "snowball effect", Zhang et al., 2023).

Because proprietary APIs are stateless black boxes that **do not expose the KV cache**, the platform cannot perform partial remediation. Detecting an error post-generation forces full response abandonment: imposing a **12–25 second latency freeze** and a **22% Hallucination Tax** that compounds linearly with query volume.

---

## 2. Repository Structure

| File | Role | Purpose |
|:---|:---|:---|
| [`01_attention_contamination_trap.py`](./01_attention_contamination_trap.py) | **The Trap (The Bad One)** | Demonstrates the default post-hoc approach. Tracks token-by-token attention weights in English corporate statutory law and runs a live **Attention Hit Counter** proving that downstream tokens physically attend to and build upon the hallucinated tokens. |
| [`02_in_flight_kv_rollback.py`](./02_in_flight_kv_rollback.py) | **The Control (The Good One)** | Demonstrates in-flight inference control. Intercepts contradiction via a discriminative cross-encoder (<25ms), performs selective KV-cache truncation, resumes from clean state, and outputs a cryptographically sealed SHA-256 audit log. |
| [`proof_of_mechanism.py`](./proof_of_mechanism.py) | **Consolidated Harness** | End-to-end operational script matching Act 4 of the case study video production plan. |

---

## 3. Empirical Verification: Two Phases

### Phase 1: Proving Attention-State Poisoning (`01_attention_contamination_trap.py`)
Run the default post-hoc harness:
```bash
python3 01_attention_contamination_trap.py
```
* **What it proves:**
  * Grounded legal premise: Statutory Company Law (Corporate Entity Formation & Director Liability).
  * Injected seed hallucination at indices 42..84 (inventing automatic joint liability for corporate directors).
  * Downstream token generation extracts multi-head attention weights across layers.
  * Real-time telemetry demonstrates that downstream legal advice (creditor enforcement, personal asset seizure) allocates **25%–60% of its attention energy directly to the hallucinated span**, triggering repeated attention hits.
  * Concludes with post-hoc evaluation demonstrating the $0.0057 / query Hallucination Tax.

### Phase 2: In-Flight Sentinel & KV Cache Truncation (`02_in_flight_kv_rollback.py`)
Run the target reference architecture harness:
```bash
python3 02_in_flight_kv_rollback.py
```
* **What it proves:**
  * Sentence 1 generates authentic legal prose and locks KV coordinates at sequence index 42.
  * Sentence 2 contradiction is intercepted by the DeBERTa cross-encoder in under 25ms.
  * Generation halts immediately. Active KV cache tensor is sliced back to sequence coordinate 42:
    $$\text{Tensor Shape Transition: } [1, 8, 84, 128] \longrightarrow [1, 8, 42, 128]$$
  * Contaminated attention keys are physically purged from VRAM.
  * Generation resumes cleanly with corrective steering: **downstream attention to the hallucinated coordinate is 0.00% by construction.**
  * Emits an immutable SHA-256 sealed JSON audit record addressing **EU AI Act Articles 14 & 15** and **ISO 42001**.

---

## 4. Production Architecture Mapping

In these local harnesses on Apple Silicon (MPS / unified memory), sequence dimensions are sliced directly in PyTorch (`dim=2`). In an enterprise distributed deployment (§6 of the specification), this operation maps directly to:
* **vLLM / SGLang RadixAttention:** Evicting the contaminated child branch in the prefix tree.
* **PagedAttention Block Deallocation:** Releasing non-contiguous physical page tables back to the allocation pool without memory churn.

---

## 5. Regulatory Grounding

* **EU AI Act Article 15:** Accuracy, robustness, and cybersecurity standards for high-risk legal AI.
* **EU AI Act Article 14:** Human oversight and inference-tier governance.
* **Legal Professional Responsibility:** SRA Principles 2 & 7, ABA Model Rule 1.1, and Bar disciplinary liability for false judicial submissions (*Mata v. Avianca*).
