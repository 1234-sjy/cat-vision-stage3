"""Accelerated staged fine tuning from a clean generic-origin 20-class checkpoint.

Always starts a NEW AdamW optimizer; this is not an exact optimizer resume.
Only train/validation data are opened. Failures propagate without a retry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, require_training_enabled
from scripts.train_expanded import (DATA_DIR, SEED, balanced_sample_weights,
    device_memory, file_sha256, load_classes, save_checkpoint, validate_records,
    verified_image_path)

GIB = 1024 ** 3
PROBE_BATCHES = (32, 64, 128, 192)
MIXUP_ALPHA = 0.0


class PetDataset(Dataset):
    """Module-level class so Windows spawn workers can pickle the dataset."""
    def __init__(self, records, paths, classes, transform):
        self.records = records
        self.paths = paths
        self.class_index = {name: index for index, name in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return (self.transform(load_image(self.paths[row['image_id']]))), \
            self.class_index[row['breed']] if row['is_cat'] else -1, int(row['is_cat'])


def seed_worker(worker_id):
    import numpy as np
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)


@contextmanager
def preserve_rng():
    """Probe transforms/dropout must not advance the actual training RNG."""
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def validate_source(payload, classes, classes_hash, manifest_hash, start_epoch):
    if payload.get('classes') != classes:
        raise ValueError('Source must preserve the frozen 20-class order')
    if payload.get('epoch') != start_epoch:
        raise ValueError('--start-epoch must equal the source checkpoint completed epoch')
    config = payload.get('config', {})
    if config.get('classes_sha256') != classes_hash:
        raise ValueError('Source class metadata fingerprint does not match')
    if config.get('train_manifest_sha256') != manifest_hash:
        raise ValueError('Source train/validation manifest fingerprint does not match')
    if config.get('initialization', {}).get('no_project_trained_initialization') is not True:
        raise ValueError('Source must certify generic ImageNet origin, with no inherited v2 weights')
    if config.get('initialization_sha256') != '704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b':
        raise ValueError('Unexpected generic ImageNet origin SHA256')
    if config.get('no_test_input') is not True:
        raise ValueError('Source must certify that training/selection used no test input')
    return config


class GpuTelemetry:
    """nvidia-smi only: the monitor thread never calls a torch/CUDA API."""
    def __init__(self, path, interval=1.):
        self.path, self.interval = Path(path), interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.context = {'phase': 'initialization', 'epoch': None, 'batch': None}
        self.thread = threading.Thread(target=self._run, name='gpu-telemetry', daemon=True)

    def set_context(self, phase, epoch=None, batch=None):
        with self.lock:
            self.context = {'phase': phase, 'epoch': epoch, 'batch': batch}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=6.)
        return False

    def _run(self):
        command = ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,power.draw',
                   '--format=csv,noheader,nounits']
        with self.path.open('x', encoding='utf-8', buffering=1) as stream:
            while not self.stop_event.is_set():
                with self.lock:
                    context = self.context.copy()
                row = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), **context}
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True,
                        timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    gpus = []
                    for line in result.stdout.strip().splitlines():
                        index, name, utilization, memory, temperature, power = [x.strip() for x in line.split(',')]
                        def number(value):
                            try:
                                return float(value)
                            except ValueError:
                                return None
                        gpus.append({'index': int(index), 'name': name,
                            'utilization_gpu_percent': number(utilization), 'memory_used_mib': number(memory),
                            'temperature_c': number(temperature), 'power_draw_w': number(power)})
                    row['gpus'] = gpus
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    row.update(diagnostic=f'{type(exc).__name__}: {exc}', monitoring_stopped=True)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    print('GPU TELEMETRY DIAGNOSTIC', json.dumps(row), flush=True)
                    return  # Diagnostics do not retry or hide a model error.
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                self.stop_event.wait(self.interval)


def mix_within_species(images,breeds,species,lam):
    permutation=torch.arange(len(images),device=images.device)
    for value in [0,1]:
        members=torch.where(species==value)[0]
        permutation[members]=members[torch.randperm(len(members),device=images.device)]
    return lam*images+(1-lam)*images[permutation],breeds[permutation]


def loss_for(model, images, breeds, species, breed_loss, species_loss):
    lam=1.0;secondary=breeds
    if MIXUP_ALPHA>0 and random.random()<0.5:
        import numpy as np
        lam=float(np.random.beta(MIXUP_ALPHA,MIXUP_ALPHA))
        images,secondary=mix_within_species(images,breeds,species,lam)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bp, sp = model(images)
        mask = breeds >= 0
        # Safe even for an all-dog minibatch; there is no CPU/GPU mask synchronization.
        targets = breeds.clamp_min(0)
        per_image = nn.functional.cross_entropy(bp, targets, reduction='none', label_smoothing=.03)
        if lam!=1.0:
            per_image=lam*per_image+(1-lam)*nn.functional.cross_entropy(bp,secondary.clamp_min(0),reduction='none',label_smoothing=.03)
        b_loss = (per_image * mask).sum() / mask.sum().clamp_min(1)
        return b_loss + .5 * species_loss(sp, species)


def forecast_peak(previous, next_batch, baseline, optimizer_bytes):
    """Reserve optimizer moments plus a 15% activation projection margin."""
    peak = previous['peak_allocated_bytes']
    return baseline + max(0, peak - baseline) * next_batch / previous['batch'] * 1.15 + optimizer_bytes


def choose_batch(rows):
    measured = [row for row in rows if row.get('status') == 'measured'
                and row.get('fits_training_memory_budget', True)]
    if not measured:
        raise RuntimeError('No batch size was demonstrated; stopping without fallback')
    return max(measured, key=lambda row: row['images_per_second'])['batch']


def probe_throughput(model, dataset, device, optimizer, breed_loss, species_loss, telemetry, out):
    """Progressive actual forward/backward, NO optimizer steps, NO OOM retry."""
    rows = []
    mode = model.training
    with preserve_rng():
        # Real transformed training images; image decode/augmentation is outside timed compute.
        samples = [dataset[i % len(dataset)] for i in range(max(PROBE_BATCHES))]
        fixed_images = torch.stack([row[0] for row in samples])
        fixed_breeds = torch.tensor([row[1] for row in samples], dtype=torch.long)
        fixed_species = torch.tensor([row[2] for row in samples], dtype=torch.long)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        free, total = torch.cuda.mem_get_info(device)
        external_bytes = total - free - torch.cuda.memory_reserved(device)
        budget = min(14 * GIB, total - max(0, external_bytes) - 512 * 1024 ** 2)
        moments = sum(p.numel() * p.element_size() * 2 for p in model.parameters())
        for batch in PROBE_BATCHES:
            if rows and forecast_peak(rows[-1], batch, baseline, moments) > budget:
                rows.append({'batch': batch, 'status': 'ramp_stopped_memory_forecast',
                    'projected_peak_with_optimizer_bytes': forecast_peak(rows[-1], batch, baseline, moments),
                    'budget_bytes': budget})
                save_json(out / 'throughput-probe.json', rows)
                break
            telemetry.set_context('throughput_probe', batch=batch)
            images = fixed_images[:batch].to(device)
            breeds, species = fixed_breeds[:batch].to(device), fixed_species[:batch].to(device)
            # Untimed warmup, including gradient allocation and kernel initialization.
            loss_for(model, images, breeds, species, breed_loss, species_loss).backward()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(3):
                loss = loss_for(model, images, breeds, species, breed_loss, species_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            row = {'batch': batch, 'status': 'measured', 'steps': 3,
                'wall_seconds': seconds, 'wall_step_seconds': seconds / 3,
                'images_per_second': 3 * batch / seconds,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'optimizer_moment_reserve_bytes': moments, 'budget_bytes': budget,
                'fits_training_memory_budget': torch.cuda.max_memory_allocated(device) + moments <= budget,
                'input': 'fixed transformed training images; decode and transfers outside timer',
                'optimizer_steps': 0}
            rows.append(row)
            save_json(out / 'throughput-probe.json', rows)
            print('PROBE', json.dumps(row), flush=True)
            del images, breeds, species, loss
            torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        model.train(mode)
    return choose_batch(rows)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def make_loader(dataset, batch, sampler=None):
    return DataLoader(dataset, batch_size=batch, sampler=sampler, shuffle=False,
        num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))


def evaluate(model, loader, device, classes, species_loss, telemetry, epoch):
    model.eval()
    count = len(classes)
    support = torch.zeros(count, dtype=torch.long, device=device)
    correct = torch.zeros_like(support)
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for step, (images, breeds, species) in enumerate(loader, 1):
            telemetry.set_context('validation', epoch, step)
            images = images.to(device, non_blocking=True)
            breeds, species = breeds.to(device, non_blocking=True), species.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                bp, sp = model(images)
                mask = breeds >= 0
                per_image = nn.functional.cross_entropy(bp, breeds.clamp_min(0), reduction='none', label_smoothing=.03)
                b_sum = (per_image * mask).sum()
                s_sum = species_loss(sp, species) * len(images)
            predicted, accepted = bp.argmax(1), sp.argmax(1) == 1
            support += torch.bincount(breeds[mask], minlength=count)
            correct += torch.bincount(breeds[mask & (predicted == breeds)], minlength=count)
            totals += torch.stack((b_sum.double(), s_sum.double(), mask.sum(),
                (mask & accepted).sum(), (~mask).sum(), (~mask & accepted).sum()))
    torch.cuda.synchronize(device)
    support, correct, totals = support.cpu().tolist(), correct.cpu().tolist(), totals.cpu().tolist()
    if not all(support):
        raise ValueError('Every breed must be represented in validation')
    accuracies = [a / n for a, n in zip(correct, support)]
    val_loss = totals[0] / sum(support) + .5 * totals[1] / len(loader.dataset)
    if not torch.isfinite(torch.tensor(val_loss)):
        raise FloatingPointError('Non-finite validation loss; stopping without retry')
    return {'val_loss': val_loss, 'val_breed_accuracy': sum(correct) / sum(support),
        'val_breed_macro_accuracy': sum(accuracies) / count,
        'val_old12_macro_accuracy': sum(accuracies[:12]) / 12,
        'val_new8_macro_accuracy': sum(accuracies[12:]) / 8,
        'val_per_breed': {name: {'count': n, 'correct': a, 'accuracy': a / n}
            for name, n, a in zip(classes, support, correct)},
        'val_cats': sum(support), 'val_cats_accepted': int(totals[3]),
        'val_dogs': int(totals[4]), 'val_false_cat_dogs': int(totals[5])}


def run(args):
    global MIXUP_ALPHA
    MIXUP_ALPHA=args.mixup
    from catvision.breed_transformer import BreedSwin, train_transform, eval_transform
    require_training_enabled()
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; stopping without CPU/precision fallback')
    device = torch.device('cuda')
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision('high')
    classes, class_zh = load_classes(args.classes)
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    classes_hash = file_sha256(args.classes)
    records = json.loads(manifest_bytes)
    validate_records(records, classes, {'train', 'val'})
    for split in ('train', 'val'):
        if {r['breed'] for r in records if r['split'] == split and r['is_cat']} != set(classes):
            raise ValueError(f'Every breed must be represented in {split}')
    payload = torch.load(args.source, map_location='cpu', weights_only=True)
    source_config = validate_source(payload, classes, classes_hash, manifest_hash, args.start_epoch)
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError('A unique output directory and checkpoint are required; no overwrite')
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for index, row in enumerate(records, 1):
        paths[row['image_id']] = verified_image_path(row, args.data_root)
        if index % 500 == 0 or index == len(records):
            print(f'INPUT CHECK {index}/{len(records)} train/val image SHA256 verified; no test', flush=True)
    train_data = PetDataset([r for r in records if r['split'] == 'train'], paths, classes, train_transform(args.image_size,args.preprocess))
    val_data = PetDataset([r for r in records if r['split'] == 'val'], paths, classes, eval_transform(args.image_size,args.preprocess))
    model = BreedSwin(pretrained_path=None, classes=classes).float()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.backbone_lr},
        {'params': list(model.breed_head.parameters()) + list(model.species_head.parameters()), 'lr': args.heads_lr}],
        weight_decay=1e-4)
    breed_loss, species_loss = nn.CrossEntropyLoss(label_smoothing=.03), nn.CrossEntropyLoss()
    config = {**source_config, 'classes': classes, 'classes_zh': class_zh,
        'source_checkpoint': str(args.source.resolve()), 'source_checkpoint_sha256': file_sha256(args.source),
        'source_completed_epoch': args.start_epoch, 'start_epoch': args.start_epoch,
        'end_epoch': args.end_epoch, 'epochs_max': args.end_epoch,
        'continuation': 'staged fine tuning with a fresh AdamW optimizer; not exact resume',
        'source_has_optimizer_state': 'optimizer_state_dict' in payload,
        'source_optimizer_state_loaded': False, 'optimizer': 'AdamW',
        'backbone_lr': args.backbone_lr, 'heads_lr': args.heads_lr, 'weight_decay': 1e-4,
        'image_size':args.image_size,'preprocess':args.preprocess,'mixup_alpha':args.mixup,
        'mixup_probability':0.5 if args.mixup>0 else 0,
        'lr_schedule': 'constant during this stage', 'warmup_frozen_epochs': 0,
        'breed_label_smoothing': .03, 'species_loss_weight': .5,
        'train_manifest_sha256': manifest_hash, 'classes_sha256': classes_hash,
        'train_manifest': str(args.manifest.resolve()), 'data_root': str(args.data_root.resolve()),
        'batch': None, 'validation_batch': 128, 'workers': 4, 'cpu_threads': 8,
        'persistent_workers': True, 'prefetch_factor': 2, 'pin_memory': True,
        'pause_after_sync_seconds': 0, 'precision': 'BF16 autocast with FP32 master weights',
        'scaler': None, 'sampling': {'cat_mass': .8, 'dog_mass': .2, 'cat_breeds_equal_mass': True,
            'replacement': True, 'samples_per_epoch': len(train_data)},
        'train_source_count': len(train_data), 'val_source_count': len(val_data),
        'previous_loader': {key: source_config.get(key) for key in
            ('batch', 'validation_batch', 'workers', 'cpu_threads', 'train_source_count', 'val_source_count', 'train_batches', 'val_batches')},
        'probe_batches': list(PROBE_BATCHES), 'probe_optimizer_steps': 0,
        'no_test_input': True, 'automatic_retry': False, 'automatic_resume': False,
        'selection': 'maximum validation mean per-breed raw-head accuracy; lower validation loss on tie',
        'seed': SEED, 'device': str(device), 'torch_version': str(torch.__version__)}
    save_json(args.output / 'config.json', config)
    del payload
    history, best_score = [], None
    started = time.perf_counter()
    with GpuTelemetry(args.output / 'gpu-telemetry.jsonl') as telemetry:
        batch = probe_throughput(model, train_data, device, optimizer, breed_loss, species_loss, telemetry, args.output)
        sampler = WeightedRandomSampler(balanced_sample_weights(train_data.records, classes),
            len(train_data), replacement=True, generator=torch.Generator().manual_seed(SEED))
        train = make_loader(train_data, batch, sampler)
        val = make_loader(val_data, 128)
        config.update(batch=batch, train_batches=len(train), val_batches=len(val))
        save_json(args.output / 'config.json', config)
        print('CONFIG', json.dumps(config), flush=True)
        for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
            telemetry.set_context('train', epoch, 0)
            model.train()
            train_started = time.perf_counter()
            loss_sum = torch.zeros((), device=device)
            train_seen = 0
            compute_start, compute_end = [], []
            for step, (images, breeds, species) in enumerate(train, 1):
                telemetry.set_context('train', epoch, step)
                images = images.to(device, non_blocking=True)
                breeds, species = breeds.to(device, non_blocking=True), species.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                loss = loss_for(model, images, breeds, species, breed_loss, species_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                end.record()
                compute_start.append(begin)
                compute_end.append(end)
                loss_sum += loss.detach() * len(images)
                train_seen += len(images)
                if step == 1 or step % 50 == 0 or step == len(train):
                    print('TRAIN', json.dumps({'epoch': epoch, 'batch': step, 'batches': len(train),
                        'loss': float(loss_sum) / train_seen, 'images_seen': train_seen,
                        'end_to_end_images_per_second_so_far': train_seen / (time.perf_counter() - train_started),
                        **device_memory(torch, device)}), flush=True)
            torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - train_started
            compute_seconds = sum(a.elapsed_time(b) for a, b in zip(compute_start, compute_end)) / 1000
            train_loss = float(loss_sum) / train_seen
            if not torch.isfinite(torch.tensor(train_loss)):
                raise FloatingPointError(f'Non-finite training loss epoch={epoch}; no retry')
            val_started = time.perf_counter()
            metrics = evaluate(model, val, device, classes, species_loss, telemetry, epoch)
            row = {'epoch': epoch, 'stage_epoch': epoch - args.start_epoch,
                'backbone_frozen': False, 'train_loss': train_loss, **metrics,
                'train_images_seen': train_seen, 'train_seconds': train_seconds,
                'train_end_to_end_images_per_second': train_seen / train_seconds,
                'train_gpu_compute_seconds': compute_seconds,
                'train_gpu_compute_images_per_second': train_seen / compute_seconds,
                'validation_seconds': time.perf_counter() - val_started,
                'learning_rates': [g['lr'] for g in optimizer.param_groups],
                'train_batches': len(train), 'val_batches': len(val),
                'seconds': time.perf_counter() - train_started,
                'elapsed_seconds': time.perf_counter() - started, **device_memory(torch, device)}
            history.append(row)
            telemetry.set_context('checkpoint', epoch)
            checkpoint = {'state_dict': cpu_tree(model.state_dict()), 'classes': classes,
                'epoch': epoch, 'stage_epoch': epoch - args.start_epoch, 'config': config,
                'metrics': row, 'optimizer_state_dict': cpu_tree(optimizer.state_dict()),
                'scaler_state_dict': None, 'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state_all': torch.cuda.get_rng_state_all(), 'python_rng_state': random.getstate(),
                'sampler_rng_state': sampler.generator.get_state()}
            save_checkpoint(args.output / f'epoch{epoch:02d}.pt', checkpoint, immutable=True)
            save_json(args.output / 'history.json', history)
            score = (row['val_breed_macro_accuracy'], -row['val_loss'])
            if best_score is None or score > best_score:
                best_score = score
                save_checkpoint(args.checkpoint, checkpoint)
                save_json(args.output / 'best.json', row)
            del checkpoint
            print('EPOCH', json.dumps(row), 'BEST', best_score, flush=True)
        save_json(args.output / 'complete.json', {
            'best': json.loads((args.output / 'best.json').read_text(encoding='utf-8')),
            'stage_epochs_completed': len(history), 'source_completed_epoch': args.start_epoch,
            'last_global_epoch': history[-1]['epoch'], 'stage_global_epochs': [r['epoch'] for r in history],
            'seconds': time.perf_counter() - started,
            'best_weights_sha256': file_sha256(args.checkpoint), 'no_test_input': True,
            'continuation': config['continuation']})
    print('ACCELERATED STAGE COMPLETED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--start-epoch', type=int, required=True, help='Completed epoch in --source')
    parser.add_argument('--end-epoch', type=int, required=True, help='Final global epoch, inclusive')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DATA_DIR / 'train-manifest.json')
    parser.add_argument('--classes', type=Path, default=DATA_DIR / 'classes.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image-size',type=int,default=224)
    parser.add_argument('--preprocess',choices=['center','letterbox'],default='center')
    parser.add_argument('--mixup',type=float,default=0)
    parser.add_argument('--backbone-lr',type=float,default=1e-5)
    parser.add_argument('--heads-lr',type=float,default=1e-4)
    args = parser.parse_args()
    if args.start_epoch < 1 or args.end_epoch <= args.start_epoch:
        parser.error('Require end-epoch > start-epoch >= 1')
    with ModelTask('train-accelerated-v3'):
        run(args)


if __name__ == '__main__':
    main()
