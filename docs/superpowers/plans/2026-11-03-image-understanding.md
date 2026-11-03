# Image Understanding Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. User authorized autonomous execution; implementation stays in the current checkout.

**Goal:** Add grounded cat-image observations and follow-up image questions while retaining the trained classifier.

**Architecture:** Classify with the frozen LoRA path; describe/answer from the same image with the adapter disabled temporarily. Serialize request-level GPU work using the existing executor and ModelTask. Keep A/B concurrency within recognition.

**Tech Stack:** Python, transformers/Qwen3-VL, PEFT, FastAPI, plain HTML/JS.

**Spec:** docs/superpowers/specs/2026-11-03-image-understanding-design.md

## Global Constraints

- No model download or new training is needed; preserve all models/data/logs/course sources.
- No static breed features or filename/other-route answer in the description prompt.
- Classification accuracy is distinct from unlabelled descriptive quality.
- Record actual demo latency without editing waiting time.
- Preserve real Git history; prepare local upload package without pushing a remote.

## Review Focus

- Questions before recognition, absent inputs, or invalid IDs must fail clearly.
- Incomplete structured observations retain raw text rather than fabricated fields.
- Answer jobs and recognition share one executor so adapter state cannot overlap.
- A new uploaded image clears old question state and answers.
- No-cat images still receive an observation of actual visible contents.

### Task 1: Observation engine and image question API/UI

- [ ] Add boundary tests for structured descriptions, original-image questions and invalid parent jobs; observe RED.
- [ ] Implement observation parsing/generation, adapter restoration, queued question endpoint and UI.
- [ ] Run focused CPU tests and existing API/concurrency suite; record result and commit.

### Task 2: Real GPU outputs and demo

- [ ] Inspect fixed validation images with visibly different appearances, a non-cat input and one question each; preserve raw outputs.
- [ ] Re-run the frozen 120-image classification regression with observations enabled; compare classification predictions to preserved results without tuning on this set.
- [ ] Record a continuous IAB upload, dual-route output and user question; preserve frames and backend results.

### Task 3: Documentation and delivery

- [ ] Explain PDF/LAVIS alignment, actual feature behavior, classification scope and observed limitations in README and plain first-person Word using humanizer-zh.
- [ ] Commit real development history and build the revised three-file handoff and complete preservation archive.
- [ ] Send revised three files and actual parameters to File Transfer Assistant; continue the authorized relay until a new user “好的”, then shut down.
