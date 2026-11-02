# Expanded breeds and parallel inference

Goal: expand beyond 12 breeds, improve the independent VLM route, and run both routes concurrently. Accuracy takes priority over latency.

Architecture: preserve the YOLO/SAM/Swin and Qwen image-input routes. Share immutable breed metadata, never predictions. One request owns two workers and two CUDA streams; loading is sequential, inference overlaps. Multiple uploaded requests remain queued to bound memory. Training is a separate lower-load activity. Normal serving keeps models loaded with no artificial pauses.

Authorization: the user explicitly requests implementation and autonomous choices. Old one-model-at-a-time rules are superseded for inference. Preserve logs, data, course requirements, all real Git history. Failures stop; no automatic retry/restart. No remote upload.

Approach: use a cleaned expanded dataset with hash/near-duplicate and animal-identity split checks; exclude coat colours presented as breeds. Select the classifier and VLM settings only on validation. Keep Oxford official test for old-class regression and freeze a new expanded test before training. Preserve the previous 12-class checkpoint as a baseline. Profile the current VLM before changing precision, input size or output format.

Tasks:
- [ ] 1. Remove the requested helper from HTML, update runtime policy, implement isolated route workers and stream-aware synchronization. Tests: concurrent workers actually meet a barrier; errors wait for sibling completion before closing engines; no answer sharing; lower-load training remains unchanged.
- [ ] 2. Acquire and audit expanded data, write dynamic class mapping and a reproducible grouped split manifest. Test duplicate/identity groups cannot cross splits; old checkpoints retain their own class order.
- [ ] 3. Profile VLM by phase, compare validation-only settings, implement the selected faster and more accurate schema. Record tokens, timings, prompt versions, errors and correct counts.
- [ ] 4. Fine-tune expanded classifier with batch 8, workers 0, CPU 2, pauses after synchronized batches; save every epoch and select by validation. Evaluate fixed old-class and expanded tests once selected. Record uncertainty/unknown handling honestly.
- [ ] 5. Measure real parallel inference, update README and the plain first-person Word, record a fresh single-image demonstration, commit real work and regenerate the three-file local handoff with Git bundle.

Review focus: class order in old checkpoints; duplicate animals across sources; runtime exceptions while the other route is still using CUDA; no-cat and ambiguous cats; correct scope for all accuracy claims.
