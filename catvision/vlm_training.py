"""LoRA training helpers; importing this module never initializes a CUDA model.

Only the assistant's class code and its end token are supervised. Source labels
are dataset labels, never predictions from another classifier.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from uuid import uuid4

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .common import load_image
from scripts.train_accelerated import GpuTelemetry

GIB = 1024 ** 3
BASE_MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
ASSISTANT_HEADER = '<|im_start|>assistant\n'
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def durable_json(path: Path, value, *, immutable=False) -> None:
    """Atomic JSON replacement, with durable writes and no immutable overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    if immutable and path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def append_json(stream, row) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def load_classes(path: Path) -> list[str]:
    from .common import BREEDS
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(rows, list) or len(rows) != 20
            or [r.get('class_index') for r in rows] != list(range(20))):
        raise ValueError('Expected the frozen ordered 20-class mapping')
    classes = [r.get('breed') for r in rows]
    if classes != list(BREEDS):
        raise ValueError('Class order differs from the project 20-class mapping')
    return classes


def validate_records(records: list[dict], classes: list[str], split: str) -> None:
    if split not in {'train', 'val'}:
        raise ValueError('Training may only open train or validation records')
    if not isinstance(records, list) or not records:
        raise ValueError('Empty or non-list image manifest')
    paths, ids, hashes = set(), set(), set()
    mapping = {name: index for index, name in enumerate(classes)}
    for row in records:
        if row.get('split') != split:
            raise ValueError(f'Forbidden image split: expected {split}')
        if type(row.get('is_cat')) is not bool:
            raise ValueError('is_cat must be a boolean')
        breed = row.get('breed')
        if row['is_cat']:
            if breed not in mapping or row.get('class_index', mapping[breed]) != mapping[breed]:
                raise ValueError('Cat label does not match the frozen class mapping')
            expected_target = breed
        else:
            if breed != 'negative_dog' or row.get('class_index', -1) != -1:
                raise ValueError('Dog negatives must retain their source label')
            expected_target = 'no_cat'
        if row.get('target_text') != expected_target:
            raise ValueError('Assistant target differs from the source dataset label')
        relative = row.get('relative_path')
        if not isinstance(relative, str) or not relative:
            raise ValueError('Missing image path')
        # Explicit Windows checks also work when CPU tests run on another OS.
        path = Path(relative)
        if path.is_absolute() or path.drive or ':' in relative or '..' in relative.replace('\\', '/').split('/'):
            raise ValueError('Image path must stay inside the data root')
        sha = row.get('sha256', '')
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid image SHA256')
        if not isinstance(row.get('group'), str) or not row['group']:
            raise ValueError('Every image needs an identity group')
        image_id = row.get('image_id', relative)
        if relative in paths or image_id in ids or sha in hashes:
            raise ValueError('Duplicate image id, path or SHA256 in manifest')
        paths.add(relative)
        ids.add(image_id)
        hashes.add(sha)


def validate_disjoint(train: list[dict], validation: list[dict]) -> None:
    for key in ('sha256', 'group', 'relative_path'):
        if {row[key] for row in train} & {row[key] for row in validation}:
            raise ValueError(f'Train/validation overlap in {key}')


def verify_frozen_manifest(path: Path, freeze: dict) -> str:
    """Check only the explicitly requested manifest, never open test records."""
    if Path(path).name not in {'train.json', 'val-selection.json', 'val-full.json', 'classes.json'}:
        raise ValueError('Unapproved training/validation metadata filename')
    expected = freeze.get('manifests', {}).get(Path(path).name, {}).get('sha256')
    if not expected or file_sha256(path) != expected:
        raise ValueError(f'Frozen manifest changed: {Path(path).name}')
    return expected


def verified_image_paths(records: list[dict], data_root: Path) -> list[Path]:
    base = Path(data_root).resolve(strict=True)
    paths = []
    for row in records:
        path = (base / row['relative_path']).resolve(strict=True)
        if not path.is_relative_to(base):
            raise ValueError('Resolved image path escapes data root')
        if file_sha256(path) != row['sha256']:
            raise ValueError(f"Image SHA256 changed: {row['relative_path']}")
        paths.append(path)
    return paths


def balanced_sample_weights(records: list[dict], classes: list[str]) -> list[float]:
    counts = Counter(row['breed'] for row in records if row['is_cat'])
    dog_count = sum(not row['is_cat'] for row in records)
    if not dog_count or any(counts[name] == 0 for name in classes):
        raise ValueError('Training needs every cat class and dog negatives')
    return [.8 / (len(classes) * counts[row['breed']]) if row['is_cat'] else .2 / dog_count
            for row in records]


class VlmDataset(Dataset):
    """Windows-spawn-safe CPU decoding; no processor or model in a worker."""
    def __init__(self, records, paths):
        self.records, self.paths = records, paths

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {'image': load_image(self.paths[index]), 'record': self.records[index]}


def identity_collate(samples):
    return samples


def seed_worker(_worker_id):
    import numpy as np
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


@contextmanager
def preserve_rng():
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def assistant_labels(input_ids, attention_mask, header_ids, end_id):
    """Find the LAST assistant header after image-token expansion.

    Everything through that header, all padding, and anything after im_end is
    ignored. This does not assume that one image placeholder is one input token.
    """
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or not header_ids:
        raise ValueError('Expected matching batch token/mask matrices and a nonempty header')
    labels = torch.full_like(input_ids, -100)
    marker = list(header_ids)
    for row_number in range(input_ids.shape[0]):
        tokens = input_ids[row_number].tolist()
        active = attention_mask[row_number].bool().tolist()
        starts = [i for i in range(len(tokens) - len(marker) + 1)
                  if tokens[i:i + len(marker)] == marker and all(active[i:i + len(marker)])]
        if not starts:
            raise ValueError('Assistant header not found in expanded model inputs')
        start = starts[-1] + len(marker)
        endings = [i for i in range(start, len(tokens)) if active[i] and tokens[i] == end_id]
        if not endings or endings[0] == start:
            raise ValueError('Assistant target needs a nonempty code and im_end token')
        end = endings[0] + 1
        if not all(active[start:end]):
            raise ValueError('Padding interrupts the assistant target')
        labels[row_number, start:end] = input_ids[row_number, start:end]
    return labels


def collate_processor(processor, samples, prompt: str, *, training: bool):
    processor.tokenizer.padding_side = 'right' if training else 'left'
    messages = []
    for sample in samples:
        conversation = [{'role': 'user', 'content': [
            {'type': 'image', 'image': sample['image']}, {'type': 'text', 'text': prompt}]}]
        if training:
            conversation.append({'role': 'assistant', 'content': [
                {'type': 'text', 'text': sample['record']['target_text']}]})
        messages.append(conversation)
    inputs = processor.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=not training, return_dict=True, return_tensors='pt', padding=True)
    if training:
        header = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
        end_id = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if end_id is None or end_id == processor.tokenizer.unk_token_id:
            raise ValueError('Tokenizer does not define the Qwen im_end token')
        inputs['labels'] = assistant_labels(inputs['input_ids'], inputs['attention_mask'], header, end_id)
    return inputs


def supervised_positions(labels):
    """Logit position i predicts token i+1, matching standard causal LM loss."""
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError('Expected a causal label matrix')
    shifted = labels[:, 1:]
    positions = torch.where((shifted != -100).any(dim=0))[0]
    if positions.numel() == 0:
        raise ValueError('No supervised assistant tokens')
    return positions, shifted[:, positions]


def assistant_only_loss(model, inputs):
    """Exactly the masked causal CE, without prompt-wide vocabulary logits.

    Qwen3-VL supports Tensor logits_to_keep, selecting hidden positions before
    its vocabulary projection. All prompt/image positions still participate in
    the decoder attention. Incompatible APIs fail directly, without fallback.
    """
    labels = inputs['labels']
    positions, targets = supervised_positions(labels)
    arguments = {key: value for key, value in inputs.items() if key != 'labels'}
    outputs = model(**arguments, logits_to_keep=positions, use_cache=False)
    logits = outputs.logits
    if logits.shape[:2] != targets.shape:
        raise RuntimeError('Model did not honor Tensor logits_to_keep')
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100)


def forecast_peak(previous, next_batch, baseline, optimizer_bytes):
    """A 15% activation margin, plus optimizer moments not allocated in probes."""
    activation = max(0, previous['peak_allocated_bytes'] - baseline)
    return baseline + activation * next_batch / previous['batch'] * 1.15 + optimizer_bytes


def choose_batch(rows):
    measured = [row for row in rows if row.get('status') == 'measured'
                and row.get('fits_training_memory_budget', False)]
    if not measured:
        raise RuntimeError('No batch fits the demonstrated training memory budget; no automatic retry')
    return max(measured, key=lambda row: row['images_per_second'])['batch']


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def rng_state():
    import numpy as np
    numpy_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {
        'algorithm': numpy_state[0], 'keys': torch.from_numpy(numpy_state[1].copy()),
        'position': numpy_state[2], 'has_gauss': numpy_state[3], 'cached_gaussian': numpy_state[4]},
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def atomic_checkpoint(path: Path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Immutable training state already exists: {path}')
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def save_adapter_snapshot(model, destination, optimizer, scheduler, *, epoch, global_step, config,
                          data_rng_state=None):
    """Adapter-only immutable directory; never copy the frozen 4B model tensors."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True, save_embedding_layers=False)
    adapter_config_path = temporary / 'adapter_config.json'
    adapter_config = json.loads(adapter_config_path.read_text(encoding='utf-8'))
    adapter_config['base_model_name_or_path'] = BASE_MODEL_ID
    durable_json(adapter_config_path, adapter_config)
    atomic_checkpoint(temporary / 'training-state.pt', {
        'epoch': epoch, 'global_step': global_step, 'config': config,
        'optimizer_state_dict': cpu_tree(optimizer.state_dict()),
        'scheduler_state_dict': scheduler.state_dict(), 'rng_state': rng_state(),
        'data_rng_state': cpu_tree(data_rng_state),
        'adapter_only': True, 'automatic_resume': False})
    # save_pretrained closes files; explicitly make its retained artifacts durable.
    for path in temporary.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    return destination


def power_status():
    if sys.platform != 'win32':
        return {'ac_line_status': 'unknown'}
    import ctypes
    from ctypes import wintypes
    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [('ACLineStatus', wintypes.BYTE), ('BatteryFlag', wintypes.BYTE),
                    ('BatteryLifePercent', wintypes.BYTE), ('SystemStatusFlag', wintypes.BYTE),
                    ('BatteryLifeTime', wintypes.DWORD), ('BatteryFullLifeTime', wintypes.DWORD)]
    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return {'ac_line_status': 'unavailable'}
    return {'ac_line_status': {0: 'offline', 1: 'online', 255: 'unknown'}.get(status.ACLineStatus, 'unknown'),
            'battery_percent': None if status.BatteryLifePercent == 255 else status.BatteryLifePercent,
            'battery_saver': bool(status.SystemStatusFlag)}


class VlmGpuTelemetry(GpuTelemetry):
    """Reuse the monitor lifecycle, adding clocks and Windows AC/battery state."""
    def _run(self):
        command = ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,'
                   'power.draw,clocks.sm,clocks.mem', '--format=csv,noheader,nounits']
        with self.path.open('x', encoding='utf-8', buffering=1) as stream:
            while not self.stop_event.is_set():
                with self.lock:
                    context = self.context.copy()
                row = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), **context, **power_status()}
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True,
                        timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    gpus = []
                    for line in result.stdout.strip().splitlines():
                        fields = [part.strip() for part in line.split(',')]
                        if len(fields) != 8:
                            raise ValueError('Unexpected nvidia-smi telemetry columns')
                        def number(value):
                            try:
                                return float(value)
                            except ValueError:
                                return None
                        gpus.append(dict(zip(('index', 'name', 'utilization_gpu_percent',
                            'memory_used_mib', 'temperature_c', 'power_draw_w', 'sm_clock_mhz',
                            'memory_clock_mhz'), [int(fields[0]), fields[1], *map(number, fields[2:])])))
                    row['gpus'] = gpus
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    row.update(diagnostic=f'{type(exc).__name__}: {exc}', monitoring_stopped=True)
                    append_json(stream, row)
                    print('GPU TELEMETRY DIAGNOSTIC', json.dumps(row), flush=True)
                    return
                append_json(stream, row)
                self.stop_event.wait(self.interval)
