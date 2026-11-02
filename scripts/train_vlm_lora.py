"""Train class-code-only Qwen3-VL LoRA from frozen source labels.

No downloads, test inputs, automatic retries, automatic resume, quantization,
CPU offload, or full-model checkpoint copies. GPU work is owned by this CLI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from uuid import uuid4

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from catvision.common import ROOT
from catvision.run_control import ModelTask, require_training_enabled
from catvision.vlm_training import (BASE_MODEL_ID, GIB, LORA_TARGETS, VlmDataset,
    VlmGpuTelemetry, append_json, assistant_only_loss, balanced_sample_weights,
    choose_batch, collate_processor, durable_json, file_sha256, forecast_peak,
    identity_collate, load_classes, preserve_rng, save_adapter_snapshot,
    seed_worker, validate_disjoint, validate_records, verified_image_paths,
    verify_frozen_manifest)

SEED = 20261103
DATA_DIR = ROOT / 'artifacts/optimization-v4/data'


def source_fingerprints(model_dir):
    extensions = {'.json', '.safetensors', '.model', '.txt', '.jinja', '.tiktoken'}
    files = sorted(path for path in Path(model_dir).rglob('*')
                   if path.is_file() and path.suffix in extensions)
    if not any(path.suffix == '.safetensors' for path in files):
        raise ValueError('Expected the local official safetensors base model')
    rows = [{'path': path.relative_to(model_dir).as_posix(), 'bytes': path.stat().st_size,
             'sha256': file_sha256(path)} for path in files]
    canonical = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'model': BASE_MODEL_ID, 'local_path': str(Path(model_dir).resolve()),
            'files': rows, 'aggregate_sha256': hashlib.sha256(canonical).hexdigest(),
            'local_files_only': True, 'project_classifier_predictions_used': False}


def train_mode(model):
    model.train()
    model.config.use_cache = False
    # Frozen vision dropout/checkpoint behavior must stay deterministic/eval.
    model.get_base_model().model.visual.eval()


def attach_lora(engine):
    from peft import LoraConfig, TaskType, get_peft_model
    for parameter in engine.model.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(engine.model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=.05, bias='none', target_modules=list(LORA_TARGETS)))
    trainable = []
    dtype_counts = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if 'lora_' not in name or '.visual.' in name:
                raise ValueError(f'Unexpected trainable non-text-adapter tensor: {name}')
            parameter.data = parameter.data.float()
            trainable.append({'name': name, 'shape': list(parameter.shape),
                              'numel': parameter.numel(), 'dtype': str(parameter.dtype)})
        elif parameter.dtype != torch.bfloat16:
            raise ValueError(f'Frozen base parameter must be BF16: {name} ({parameter.dtype})')
        dtype_counts[str(parameter.dtype)] = dtype_counts.get(str(parameter.dtype), 0) + parameter.numel()
    if not trainable:
        raise ValueError('No LoRA parameters were attached')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    engine.model = model  # LocalVLM.close must release the adapter-wrapped model too.
    train_mode(model)
    return {'trainable_parameters': trainable,
            'trainable_numel': sum(row['numel'] for row in trainable),
            'total_numel': sum(parameter.numel() for parameter in model.parameters()),
            'parameter_dtype_numel': dtype_counts, 'base_storage_dtype': 'torch.bfloat16',
            'adapter_training_dtype': 'torch.float32', 'gradient_checkpointing_reentrant': False,
            'frozen_vision_mode': 'eval', 'r': 8, 'alpha': 16, 'dropout': .05,
            'bias': 'none', 'target_modules': list(LORA_TARGETS)}


def move_inputs(inputs, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()}


def checked_loss(model, inputs):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = assistant_only_loss(model, inputs)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError('Nonfinite assistant-only training loss; no retry')
    return loss


def probe_throughput(engine, dataset, optimizer, telemetry, output, prompt):
    """Real input forward/backward ramp, with no optimizer step or OOM retry."""
    rows = []
    device = engine.model.device
    trainable = [parameter for parameter in engine.model.parameters() if parameter.requires_grad]
    # FP32 AdamW first and second moments; gradients already appear in measured peaks.
    optimizer_bytes = sum(parameter.numel() * parameter.element_size() * 2 for parameter in trainable)
    with preserve_rng():
        # Select long actual processed inputs to avoid probing only a tiny image.
        candidates = []
        for index in range(min(32, len(dataset))):
            sample = dataset[index]
            inputs = collate_processor(engine.processor, [sample], prompt, training=True)
            candidates.append((int(inputs['attention_mask'].sum()), sample))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        samples = [pair[1] for pair in candidates[:16]]
        if len(samples) < 16:
            raise ValueError('The throughput probe needs sixteen actual training images')
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        free, total = torch.cuda.mem_get_info(device)
        external = max(0, total - free - torch.cuda.memory_reserved(device))
        # At least 1 GiB physical headroom, even on a 16 GiB device.
        budget = min(int(14.5 * GIB), total - external - GIB)
        durable_json(output / 'probe-inputs.json', {
            'candidate_count': len(candidates), 'selection': 'longest processed inputs among first 32 train images',
            'records': [{'relative_path': sample['record']['relative_path'],
                         'sha256': sample['record']['sha256']} for sample in samples],
            'baseline_allocated_bytes': baseline, 'optimizer_future_state_bytes': optimizer_bytes,
            'device_total_bytes': total, 'external_gpu_bytes': external,
            'training_budget_bytes': budget, 'physical_headroom_bytes': GIB}, immutable=True)
        for batch in (1, 2, 4, 8, 16):
            if rows:
                projected = forecast_peak(rows[-1], batch, baseline, optimizer_bytes)
                if projected > budget:
                    rows.append({'batch': batch, 'status': 'ramp_stopped_memory_forecast',
                                 'projected_peak_with_optimizer_bytes': projected, 'budget_bytes': budget})
                    durable_json(output / 'throughput-probe.json', rows)
                    break
            telemetry.set_context('throughput_probe', batch=batch)
            inputs = move_inputs(collate_processor(engine.processor, samples[:batch], prompt,
                                                  training=True), device)
            torch.cuda.empty_cache()
            # Warmup is also real work; a failure propagates without a retry.
            loss = checked_loss(engine.model, inputs)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            del loss
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            losses = []
            for _ in range(2):
                loss = checked_loss(engine.model, inputs)
                losses.append(float(loss.detach()))
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
                del loss
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            future_peak = max(peak, reserved) + optimizer_bytes
            row = {'batch': batch, 'status': 'measured', 'iterations': 2,
                   'seconds': elapsed, 'images_per_second': 2 * batch / elapsed,
                   'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
                   'projected_peak_with_optimizer_bytes': future_peak, 'budget_bytes': budget,
                   'fits_training_memory_budget': future_peak <= budget,
                   'input_tokens': int(inputs['input_ids'].shape[1]),
                   'supervised_tokens': int((inputs['labels'] != -100).sum()), 'losses': losses,
                   'optimizer_steps': 0}
            rows.append(row)
            durable_json(output / 'throughput-probe.json', rows)
            print('VLM PROBE', json.dumps(row), flush=True)
            del inputs
            if not row['fits_training_memory_budget']:
                break
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return rows


def validation_metrics(results, classes):
    per_class = {name: {'correct': 0, 'total': 0} for name in classes}
    dogs = {'total': 0, 'false_cat': 0, 'correct_no_cat': 0, 'parse_error': 0}
    for row in results:
        result = row['prediction']
        if row['is_cat']:
            metric = per_class[row['target_text']]
            metric['total'] += 1
            metric['correct'] += int(result['status'] == 'ok' and result['breed'] == row['target_text'])
        else:
            dogs['total'] += 1
            dogs['false_cat'] += int(result['has_cat'] is True)
            dogs['correct_no_cat'] += int(result['status'] == 'no_cat')
            dogs['parse_error'] += int(result['status'] == 'parse_error')
    if any(row['total'] == 0 for row in per_class.values()) or dogs['total'] == 0:
        raise ValueError('Validation must contain every cat class and dog negatives')
    for row in per_class.values():
        row['accuracy'] = row['correct'] / row['total']
    correct = sum(row['correct'] for row in per_class.values())
    total = sum(row['total'] for row in per_class.values())
    return {'cat_correct': correct, 'cat_total': total, 'cat_accuracy': correct / total,
            'cat_macro_accuracy': sum(row['accuracy'] for row in per_class.values()) / len(classes),
            'per_class': per_class, 'dog': {**dogs, 'false_cat_rate': dogs['false_cat'] / dogs['total'],
                'no_cat_accuracy': dogs['correct_no_cat'] / dogs['total']}}


def evaluate(engine, loader, classes, telemetry, *, epoch, prompt, prompt_version, provenance):
    from catvision.vlm_format import parse_breed_lines
    model = engine.model
    model.eval()
    model.config.use_cache = True
    results = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, samples in enumerate(loader, 1):
            telemetry.set_context('validation', epoch, batch_number)
            inputs = move_inputs(collate_processor(engine.processor, samples, prompt, training=False), model.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated = model.generate(**inputs, max_new_tokens=24, do_sample=False, use_cache=True)
            answers = generated[:, inputs['input_ids'].shape[1]:]
            texts = engine.processor.batch_decode(answers, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            for sample, text in zip(samples, texts):
                row = sample['record']
                results.append({'relative_path': row['relative_path'], 'sha256': row['sha256'],
                    'group': row['group'], 'is_cat': row['is_cat'], 'target_text': row['target_text'],
                    'prediction': parse_breed_lines(text), 'prompt_version': prompt_version,
                    'input_tokens': int(inputs['input_ids'].shape[1])})
            del inputs, generated, answers
    torch.cuda.synchronize(model.device)
    metrics = validation_metrics(results, classes)
    return {'epoch': epoch, 'prompt': prompt, 'prompt_version': prompt_version,
            'provenance': provenance, 'selection': 'cat accuracy; dog false-cat rate only breaks ties',
            'max_new_tokens': 24, 'do_sample': False, 'metrics': metrics,
            'seconds': time.perf_counter() - started, 'records': results, 'no_test_input': True}


def is_better(metrics, best):
    return best is None or (metrics['cat_accuracy'], -metrics['dog']['false_cat_rate']) > (
        best['cat_accuracy'], -best['dog']['false_cat_rate'])


def publish_selected_adapter(source, destination, metadata):
    """Publish only the selected adapter, with a runtime-provided base model."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
    temporary.mkdir()
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'README.md'):
        path = Path(source) / name
        if path.exists():
            shutil.copy2(path, temporary / name)
    if not (temporary / 'adapter_model.safetensors').is_file():
        raise ValueError('Selected epoch has no adapter safetensors')
    durable_json(temporary / 'training-provenance.json', metadata, immutable=True)
    for path in temporary.iterdir():
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)


def run(args):
    import numpy as np
    from transformers import get_cosine_schedule_with_warmup
    from catvision.vlm import LocalVLM
    from catvision.vlm_prompts import VLM_CLASS_PROMPT, VLM_CLASS_PROMPT_VERSION

    require_training_enabled()
    if args.output.exists() or args.adapter_output.exists():
        raise FileExistsError('A new training directory and new adapter directory are required; no overwrite')
    if args.epochs < 1 or args.effective_batch != 16 or args.workers < 0 or args.eval_batch_size < 1:
        raise ValueError('Expected positive epochs/eval batch, nonnegative workers and effective batch 16')
    if args.validation_manifest.name not in {'val-selection.json', 'val-full.json'}:
        raise ValueError('Only frozen validation manifests may be used for training selection')
    args.output.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    freeze_path = args.manifest_dir / 'freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    train_path, classes_path = args.manifest_dir / 'train.json', args.manifest_dir / 'classes.json'
    hashes = {path.name: verify_frozen_manifest(path, freeze)
              for path in (train_path, args.validation_manifest, classes_path)}
    classes = load_classes(classes_path)
    train_rows = json.loads(train_path.read_text(encoding='utf-8'))
    validation_rows = json.loads(args.validation_manifest.read_text(encoding='utf-8'))
    validate_records(train_rows, classes, 'train')
    validate_records(validation_rows, classes, 'val')
    validate_disjoint(train_rows, validation_rows)
    print(f'Verifying image SHA256 once: train={len(train_rows)} val={len(validation_rows)}', flush=True)
    train_data = VlmDataset(train_rows, verified_image_paths(train_rows, args.data_root))
    validation_data = VlmDataset(validation_rows, verified_image_paths(validation_rows, args.data_root))
    source = source_fingerprints(args.model_dir)
    config = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED, 'classes': classes, 'source': source, 'manifest_sha256': hashes,
        'code_sha256': {name: file_sha256(ROOT / name) for name in (
            'scripts/train_vlm_lora.py', 'catvision/vlm_training.py', 'catvision/vlm.py',
            'catvision/vlm_acceleration.py', 'catvision/vlm_prompts.py', 'catvision/run_control.py')},
        'versions': {'python': sys.version, 'torch': torch.__version__,
            **{package: version(package) for package in ('transformers', 'peft', 'accelerate')}},
        'freeze_sha256': file_sha256(freeze_path), 'data_root': str(args.data_root.resolve()),
        'training_records': len(train_rows), 'validation_records': len(validation_rows),
        'validation_manifest': str(args.validation_manifest.resolve()), 'prompt': VLM_CLASS_PROMPT,
        'prompt_version': VLM_CLASS_PROMPT_VERSION, 'supervision': 'assistant class code and im_end only',
        'loss': 'exact masked causal CE with Tensor logits_to_keep supervised-position union',
        'input_image_pixels_max': args.max_pixels, 'patch_projection': 'equivalent_linear',
        'epochs_planned': args.epochs, 'learning_rate': args.lr, 'weight_decay': .01,
        'warmup_ratio': .03, 'lr_schedule': 'cosine', 'gradient_clip_norm': 1.,
        'effective_batch': args.effective_batch, 'sampling': '80% cats, equal cat classes; 20% dog negatives',
        'cpu_workers': args.workers, 'cpu_threads': 8, 'eval_batch_size': args.eval_batch_size,
        'precision': 'BF16 AMP; FP32 LoRA master weights and AdamW states',
        'max_new_tokens': 24, 'no_test_input': True, 'no_classifier_answers': True,
        'automatic_retry': False, 'automatic_resume': False, 'automatic_restart': False,
        'optimizer': 'AdamW foreach=False', 'save_steps': args.save_steps}
    durable_json(args.output / 'config.json', config, immutable=True)

    engine = None
    with VlmGpuTelemetry(args.output / 'gpu-telemetry.jsonl', interval=1.) as telemetry:
        try:
            telemetry.set_context('base_model_loading')
            engine = LocalVLM(args.model_dir, max_pixels=args.max_pixels,
                prompt=VLM_CLASS_PROMPT, prompt_version=VLM_CLASS_PROMPT_VERSION,
                max_new_tokens=24, output_format='lines', adapter_dir=False)
            torch.set_num_threads(8)  # LocalVLM's inference policy may set a lower default.
            config['lora'] = attach_lora(engine)
            parameters = [parameter for parameter in engine.model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01, foreach=False)
            rows = probe_throughput(engine, train_data, optimizer, telemetry, args.output, VLM_CLASS_PROMPT)
            measured_batch = choose_batch(rows)
            if args.microbatch != 'auto':
                measured_batch = int(args.microbatch)
                if not any(row.get('batch') == measured_batch and row.get('status') == 'measured'
                           and row.get('fits_training_memory_budget') for row in rows):
                    raise RuntimeError('Explicit microbatch did not demonstrate a safe memory budget')
            accumulation = args.effective_batch // measured_batch
            config.update(microbatch=measured_batch, gradient_accumulation=accumulation,
                          throughput_probe=rows, probe_rng_restored=True)
            durable_json(args.output / 'config.json', config)
            if args.probe_only:
                durable_json(args.output / 'probe-complete.json', {'microbatch': measured_batch,
                    'config': config, 'optimizer_steps': 0, 'no_test_input': True}, immutable=True)
                print(f'PROBE COMPLETE; selected microbatch={measured_batch}; optimizer_steps=0', flush=True)
                return

            sampler_generator = torch.Generator().manual_seed(SEED)
            loader_generator = torch.Generator().manual_seed(SEED + 1)
            sampler = WeightedRandomSampler(balanced_sample_weights(train_rows, classes),
                num_samples=len(train_rows), replacement=True, generator=sampler_generator)
            loader_args = {'batch_size': measured_batch, 'num_workers': args.workers,
                'collate_fn': identity_collate, 'worker_init_fn': seed_worker, 'generator': loader_generator,
                'persistent_workers': args.workers > 0}
            if args.workers:
                loader_args['prefetch_factor'] = 2
            train_loader = DataLoader(train_data, sampler=sampler, **loader_args)
            # Decode validation in the main process so at most four worker processes stay resident.
            val_loader = DataLoader(validation_data, batch_size=args.eval_batch_size,
                num_workers=0, collate_fn=identity_collate, shuffle=False,
                generator=torch.Generator().manual_seed(SEED + 2))
            steps_per_epoch = math.ceil(len(train_rows) / args.effective_batch)
            total_steps = steps_per_epoch * args.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                num_warmup_steps=max(1, math.ceil(total_steps * .03)), num_training_steps=total_steps)
            config.update(optimizer_steps_per_epoch=steps_per_epoch, optimizer_steps_planned=total_steps)
            durable_json(args.output / 'config.json', config)
            history, best_metrics, best_epoch = [], None, None
            global_step, microstep = 0, 0
            started = time.perf_counter()
            with (args.output / 'train-progress.jsonl').open('x', encoding='utf-8', buffering=1) as progress:
                for epoch in range(1, args.epochs + 1):
                    train_mode(engine.model)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats(engine.model.device)
                    epoch_started = window_started = time.perf_counter()
                    epoch_sampler_start = sampler_generator.get_state()
                    window_images, window_loss, epoch_loss, epoch_images = 0, 0., 0., 0
                    for batch_index, samples in enumerate(train_loader):
                        telemetry.set_context('training', epoch, batch_index + 1)
                        inputs = move_inputs(collate_processor(engine.processor, samples, VLM_CLASS_PROMPT,
                                                              training=True), engine.model.device)
                        loss = checked_loss(engine.model, inputs)
                        loss_number = float(loss.detach())
                        # Last partial group uses its actual example count, retaining effective batch 16.
                        group_start = (batch_index // accumulation) * accumulation * measured_batch
                        group_examples = min(args.effective_batch, len(train_rows) - group_start)
                        (loss * (len(samples) / group_examples)).backward()
                        epoch_loss += loss_number * len(samples)
                        epoch_images += len(samples)
                        window_loss += loss_number * len(samples)
                        window_images += len(samples)
                        microstep += 1
                        boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
                        if boundary:
                            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            if args.save_steps and global_step % args.save_steps == 0:
                                telemetry.set_context('step_checkpoint', epoch, batch_index + 1)
                                save_adapter_snapshot(engine.model, args.output / f'step{global_step:06d}',
                                    optimizer, scheduler, epoch=epoch, global_step=global_step, config=config,
                                    data_rng_state={'sampler_generator': sampler_generator.get_state(),
                                        'loader_generator': loader_generator.get_state(),
                                        'epoch_sampler_start': epoch_sampler_start,
                                        'epoch_batches_completed': batch_index + 1, 'microstep': microstep})
                        if microstep % 20 == 0 or batch_index + 1 == len(train_loader):
                            torch.cuda.synchronize(engine.model.device)
                            elapsed = time.perf_counter() - window_started
                            row = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'epoch': epoch,
                                'microstep': microstep, 'global_step': global_step, 'epoch_batch': batch_index + 1,
                                'loss': window_loss / window_images, 'images_per_second': window_images / elapsed,
                                'window_images': window_images, 'learning_rate': scheduler.get_last_lr()[0],
                                'peak_allocated_gib': torch.cuda.max_memory_allocated(engine.model.device) / GIB,
                                'peak_reserved_gib': torch.cuda.max_memory_reserved(engine.model.device) / GIB}
                            append_json(progress, row)
                            print('VLM TRAIN', json.dumps(row), flush=True)
                            window_started, window_images, window_loss = time.perf_counter(), 0, 0.
                        del inputs, loss
                    telemetry.set_context('epoch_checkpoint', epoch)
                    checkpoint = save_adapter_snapshot(engine.model, args.output / f'epoch{epoch:02d}',
                        optimizer, scheduler, epoch=epoch, global_step=global_step, config=config,
                        data_rng_state={'sampler_generator': sampler_generator.get_state(),
                            'loader_generator': loader_generator.get_state(),
                            'epoch_sampler_start': epoch_sampler_start,
                            'epoch_batches_completed': len(train_loader), 'microstep': microstep})
                    adapter_hash = file_sha256(checkpoint / 'adapter_model.safetensors')
                    evaluated = evaluate(engine, val_loader, classes, telemetry, epoch=epoch,
                        prompt=VLM_CLASS_PROMPT, prompt_version=VLM_CLASS_PROMPT_VERSION,
                        provenance={'source_aggregate_sha256': source['aggregate_sha256'],
                            'adapter_sha256': adapter_hash, 'manifest_sha256': hashes,
                            'checkpoint': str(checkpoint.resolve())})
                    durable_json(checkpoint / 'validation.json', evaluated, immutable=True)
                    metrics = evaluated['metrics']
                    history.append({'epoch': epoch, 'global_step': global_step,
                        'train_loss': epoch_loss / epoch_images, 'train_images': epoch_images,
                        'seconds': time.perf_counter() - epoch_started, 'validation': metrics,
                        'checkpoint': str(checkpoint.resolve()), 'adapter_sha256': adapter_hash})
                    if is_better(metrics, best_metrics):
                        best_metrics, best_epoch = metrics, epoch
                    durable_json(args.output / 'history.json', history)
                    durable_json(args.output / 'best.json', {'epoch': best_epoch,
                        'metrics': best_metrics, 'selection': 'validation cat accuracy, dog false-cat tie break'})
                    print('VLM EPOCH', json.dumps(history[-1]), flush=True)
            best_checkpoint = args.output / f'epoch{best_epoch:02d}'
            complete = {'state': 'complete', 'epochs_completed': args.epochs, 'global_step': global_step,
                'best_epoch': best_epoch, 'best_validation': best_metrics,
                'best_adapter_sha256': file_sha256(best_checkpoint / 'adapter_model.safetensors'),
                'adapter_output': str(args.adapter_output.resolve()), 'config': config,
                'seconds': time.perf_counter() - started, 'no_test_input': True,
                'no_classifier_answers': True, 'automatic_retry': False}
            telemetry.set_context('publish_adapter', best_epoch)
            publish_selected_adapter(best_checkpoint, args.adapter_output, complete)
            durable_json(args.output / 'complete.json', complete, immutable=True)
            print('VLM COMPLETE', json.dumps({key: value for key, value in complete.items() if key != 'config'}),
                  flush=True)
        finally:
            if engine is not None:
                engine.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--validation-manifest', type=Path, default=None)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent.parent / 'work/cat-data')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'weights/qwen3-vl-4b')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/optimization-v4/training')
    parser.add_argument('--adapter-output', type=Path, default=ROOT / 'weights/qwen-cat-lora-v4')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--microbatch', choices=('auto', '1', '2', '4', '8', '16'), default='auto')
    parser.add_argument('--effective-batch', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-pixels', type=int, default=512 * 512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-steps', type=int, default=200,
                        help='Immutable adapter/optimizer snapshot interval; 0 disables step snapshots')
    parser.add_argument('--probe-only', action='store_true', help='Forward/backward only; no optimizer step')
    args = parser.parse_args(argv)
    if args.validation_manifest is None:
        args.validation_manifest = args.manifest_dir / 'val-selection.json'
    if args.max_pixels < 128 * 128 or args.lr <= 0 or args.save_steps < 0:
        parser.error('Expected max-pixels>=16384, lr>0 and save-steps>=0')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    with ModelTask('train-vlm-lora-v4'):
        run(arguments)
