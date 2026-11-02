# Qwen cat adaptation implementation plan

Goal: improve the independent Qwen route after its 67/120 expanded-class result, while retaining the Swin route, original data, all failures and real development history.

Architecture: local Qwen3-VL-4B BF16 base, frozen visual encoder, text-layer rank-8 LoRA, assistant-code-only supervision. Train only on the frozen source training labels. Select with the fixed validation subset; then evaluate once on the fresh held-out subset and report old shared-120 only as regression.

User has authorized autonomous optimization and GPU training. Existing trained models and old packages remain preserved. One heavy GPU activity at a time; full-speed inference still uses concurrent routes. Failures stop with logs and do not automatically restart.

- [x] Reproduce the domain gap with the original 120 inputs: current 85/120 versus historical 87/120.
- [x] Prepare 5156 source-labeled training images and fixed 250-image selection validation; retain all prior test exclusions with identity propagation. Fresh test has 1514 cats across 20 classes.
- [x] Add PEFT dependency and versioned short single-code prompt.
- [ ] Review CPU-tested training entry point and loss masking, selected-logit equivalence, data isolation, atomic adapter-only snapshots and telemetry.
- [ ] Sequential GPU smoke/probe with microbatches 1/2/4, choose measured safe throughput. No downloads overlap GPU work.
- [ ] Initial two-epoch adaptation, keep per-epoch state and validation predictions. Further changes depend only on train/validation evidence.
- [x] Implement portable adapter loading, missing-adapter readiness rejection and adapter-aware prediction fingerprints. Three focused deployment tests passed.
- [ ] Activate validation-selected adapter and freeze final configuration before evaluating fresh test; keep previous 120 as regression, with honest denominator and subgroup metrics.
- [ ] Update plain Word and repository method/results, record fresh single-image live demo, commit real history, and regenerate three-file handoff plus complete data-preserving archive.

Review focus: never train on the test subset; no classifier answers as teacher; all prompt/image/pad tokens masked; never project the entire prompt into the 150K vocabulary during training; adapters must stay portable; no full 4B FP32 conversion; no silent retry or cache-substituted demo.
