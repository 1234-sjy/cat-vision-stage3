"""Train the frozen 20-breed manifest, at the explicitly limited training load.

Only train/validation inputs are opened. There is no download, resume or retry.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, gpu_checkpoint, require_training_enabled

OLD_CLASSES = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
               'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
               'Siamese', 'Sphynx']
NEW_CLASSES = ['American_Bobtail', 'American_Curl', 'American_Shorthair',
               'Exotic_Shorthair', 'Manx', 'Norwegian_Forest', 'Scottish_Fold',
               'Turkish_Angora']
DATA_DIR = ROOT / 'artifacts/optimization-v3/data'
SEED = 20261102


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_classes(path: Path) -> tuple[list[str], list[str]]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError('Expected the frozen 20-class list')
    if [r['class_index'] for r in rows] != list(range(20)):
        raise ValueError('Class indices must be contiguous and ordered')
    names = [r['breed'] for r in rows]
    if names != OLD_CLASSES + NEW_CLASSES:
        raise ValueError('Unexpected expanded class order; preserve original 12 rows')
    return names, [r['breed_zh'] for r in rows]


def validate_records(records: list[dict], classes: list[str], allowed_splits: set[str]) -> None:
    if not records:
        raise ValueError('Empty manifest')
    ids, hashes, groups, paths = set(), {}, {}, set()
    mapping = {name: index for index, name in enumerate(classes)}
    for row in records:
        if row['split'] not in allowed_splits:
            raise ValueError(f"Forbidden split {row['split']}; allowed={sorted(allowed_splits)}")
        if type(row['is_cat']) is not bool:
            raise ValueError('is_cat must be boolean')
        name = row['breed']
        expected = mapping.get(name, -1)
        if (row['is_cat'] and name not in mapping) or (not row['is_cat'] and name != 'negative_dog'):
            raise ValueError(f'Unknown class/species mapping: {name}')
        if row.get('class_index', expected) != expected:
            raise ValueError('Manifest class index does not match classes.json')
        path = Path(row['relative_path'])
        if path.is_absolute() or path.drive or '..' in path.parts:
            raise ValueError('Manifest path must stay inside --data-root')
        image_id = row['image_id']
        if image_id in ids or row['relative_path'] in paths:
            raise ValueError('Duplicate manifest image id or path')
        ids.add(image_id)
        paths.add(row['relative_path'])
        if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('Invalid image SHA256')
        for key, registry in [(row['sha256'], hashes), (row['group'], groups)]:
            if key in registry and registry[key] != row['split']:
                raise ValueError('Found cross-split SHA256 or identity group')
            registry[key] = row['split']


def verified_image_path(row: dict, data_root: Path, *, verify_hash: bool = True) -> Path:
    base = data_root.resolve()
    path = (base / row['relative_path']).resolve(strict=True)
    if not path.is_relative_to(base):
        raise ValueError('Resolved image escapes --data-root')
    if verify_hash and file_sha256(path) != row['sha256']:
        raise ValueError(f"Image fingerprint changed: {row['relative_path']}")
    return path


def balanced_sample_weights(records: list[dict], classes: list[str]) -> list[float]:
    counts = Counter(row['breed'] for row in records if row['is_cat'])
    dog_count = sum(not row['is_cat'] for row in records)
    if not dog_count or any(counts[name] == 0 for name in classes):
        raise ValueError('Training requires every cat breed and dog negatives')
    return [.8 / (len(classes) * counts[row['breed']]) if row['is_cat'] else .2 / dog_count
            for row in records]


def initialize_from_v2(model, payload: dict, classes: list[str]) -> dict:
    """Copy the trained backbone/species head; map old breed rows by name."""
    import torch
    old_names = payload['classes']
    if old_names != OLD_CLASSES:
        raise ValueError('Initialization must be the trained stage2 original-12 checkpoint')
    if len(set(classes)) != len(classes) or not set(old_names).issubset(classes):
        raise ValueError('Target classes must include each original breed exactly once')
    source, target = payload['state_dict'], model.state_dict()
    if set(source) != set(target):
        raise ValueError('Initialization architecture/state keys do not match')
    head_keys = {'breed_head.weight', 'breed_head.bias'}
    for key in source:
        if key not in head_keys:
            if source[key].shape != target[key].shape:
                raise ValueError(f'Incompatible source tensor: {key}')
            target[key] = source[key].detach().to(dtype=target[key].dtype).clone()
    if source['breed_head.weight'].shape != (12, target['breed_head.weight'].shape[1]):
        raise ValueError('Expected exactly 12 source breed-head rows')
    if source['breed_head.bias'].shape != (12,):
        raise ValueError('Expected exactly 12 source breed-head biases')
    copied = []
    with torch.no_grad():
        for old_index, name in enumerate(old_names):
            new_index = classes.index(name)
            for key in head_keys:
                target[key][new_index].copy_(source[key][old_index])
            copied.append({'breed': name, 'source_row': old_index, 'target_row': new_index})
    model.load_state_dict(target, strict=True)
    model.float()
    return {'backbone': 'all tensors copied from trained stage2 checkpoint',
            'species_head': 'all tensors copied from trained stage2 checkpoint',
            'copied_breed_rows': copied,
            'new_breed_rows': [name for name in classes if name not in old_names],
            'new_row_initialization': 'torch.nn.Linear default, seeded before model construction',
            'master_weights': 'FP32'}


def save_checkpoint(path: Path, payload: dict, *, immutable: bool = False) -> None:
    import torch
    if immutable and path.exists():
        raise FileExistsError(f'Immutable epoch checkpoint retained: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as output:
        torch.save(payload, output)
        output.flush()
        os.fsync(output.fileno())
    if immutable and path.exists():
        raise FileExistsError(f'Immutable checkpoint appeared during write: {path}')
    temporary.replace(path)


def device_memory(torch, device) -> dict:
    if device.type != 'cuda':
        return {'device': 'cpu', 'allocated_mb': 0, 'reserved_mb': 0}
    return {'device': torch.cuda.get_device_name(device),
            'allocated_mb': round(torch.cuda.memory_allocated(device) / 1024**2, 1),
            'reserved_mb': round(torch.cuda.memory_reserved(device) / 1024**2, 1),
            'peak_allocated_mb': round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)}


def run(args) -> None:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from catvision.breed_transformer import BreedSwin, train_transform, eval_transform

    require_training_enabled()
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(2)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; stopping without fallback')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    amp = device.type == 'cuda'
    if amp:
        torch.cuda.manual_seed_all(SEED)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 is unavailable; no automatic precision fallback')
    torch.set_float32_matmul_precision('high')
    classes, class_zh = load_classes(args.classes)
    manifest_bytes = args.manifest.read_bytes()
    records = json.loads(manifest_bytes)
    validate_records(records, classes, {'train', 'val'})
    if {row['split'] for row in records} != {'train', 'val'}:
        raise ValueError('Both training and validation splits are required')
    for split in ['train', 'val']:
        if set(row['breed'] for row in records if row['split'] == split and row['is_cat']) != set(classes):
            raise ValueError(f'Every breed must be represented in {split}')
    out = ROOT / 'artifacts/optimization-v3/training-generic'
    destination = ROOT / 'weights/breed-swin-v3-generic.pt'
    if (out.exists() and any(out.iterdir())) or destination.exists():
        raise FileExistsError('Existing v3 training artifacts/weights retained; no overwrite/resume')
    source = ROOT / 'weights/swin_t-704ceda3.pth'
    source_sha = file_sha256(source)
    if not source_sha.startswith('704ceda3'):
        raise ValueError('Unexpected generic ImageNet initialization weights')
    out.mkdir(parents=True, exist_ok=True)
    print(f'INPUT CHECK train+val={len(records)}; no test input; source_sha256={source_sha}', flush=True)
    paths = {}
    for index, row in enumerate(records):
        paths[row['image_id']] = verified_image_path(row, args.data_root)
        if (index + 1) % 500 == 0 or index + 1 == len(records):
            print(f'INPUT CHECK {index + 1}/{len(records)} verified', flush=True)

    class PetDataset(Dataset):
        def __init__(self, split):
            self.records = [row for row in records if row['split'] == split]
            self.transform = train_transform() if split == 'train' else eval_transform()

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            row = self.records[index]
            image = load_image(paths[row['image_id']])
            return (self.transform(image), classes.index(row['breed']) if row['is_cat'] else -1,
                    int(row['is_cat']))

    train_data, val_data = PetDataset('train'), PetDataset('val')
    sampler = WeightedRandomSampler(balanced_sample_weights(train_data.records, classes),
                                    num_samples=len(train_data), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))
    train = DataLoader(train_data, batch_size=8, sampler=sampler, num_workers=0, pin_memory=False)
    val = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    model = BreedSwin(pretrained_path=source, classes=classes)
    initialization = {'source':'generic ImageNet torchvision Swin-Tiny',
        'breed_head':'20 rows freshly initialized with fixed seed',
        'species_head':'freshly initialized with fixed seed',
        'no_project_trained_initialization':True,
        'reason':'Prevent inherited exposure to heldout duplicate groups from the previous project checkpoint'}
    model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1.5e-5, 'base_lr': 1.5e-5},
        {'params': list(model.breed_head.parameters()) + list(model.species_head.parameters()),
         'lr': 5e-4, 'base_lr': 5e-4}], weight_decay=1e-4)
    breed_loss = nn.CrossEntropyLoss(label_smoothing=.03)
    species_loss = nn.CrossEntropyLoss()
    config = {'architecture': 'torchvision.swin_t', 'classes': classes, 'classes_zh': class_zh,
              'initialization_file': source.relative_to(ROOT).as_posix(),
              'initialization_sha256': source_sha, 'initialization': initialization,
              'classes_sha256': file_sha256(args.classes),
              'train_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'train_manifest': str(args.manifest.resolve()), 'data_root': str(args.data_root.resolve()),
              'seed': SEED, 'epochs_max': args.epochs, 'warmup_frozen_epochs': 2,
              'patience': args.patience, 'early_stop_minimum_epoch': 10,
              'batch': 8, 'validation_batch': 8, 'workers': 0, 'cpu_threads': 2, 'image_size': 224,
              'backbone_lr': 1.5e-5, 'heads_lr': 5e-4, 'optimizer': 'AdamW', 'weight_decay': 1e-4,
              'lr_schedule': 'cosine, factor 1.0 to 0.15 after two frozen epochs',
              'breed_label_smoothing': .03, 'species_loss_weight': .5,
              'precision': 'BF16 autocast with FP32 master weights' if amp else 'FP32 CPU',
              'tf32_matmul_allowed': True, 'pause_after_sync_seconds': .05,
              'sampling': {'replacement': True, 'samples_per_epoch': len(train_data),
                           'cat_mass': .8, 'dog_mass': .2, 'cat_breeds_equal_mass': True},
              'source_images': 'original images only; no subject crop',
              'train_source_count': len(train_data), 'val_source_count': len(val_data),
              'train_batches': len(train), 'val_batches': len(val),
              'selection': 'maximum validation mean per-breed raw-head accuracy; lower validation loss on tie',
              'cat_threshold': .5, 'tta': False, 'no_test_input': True,
              'automatic_retry': False, 'automatic_resume': False,
              'torch_version': str(torch.__version__), 'device': str(device)}
    save_json(out / 'config.json', config)
    print('CONFIG', json.dumps(config), flush=True)
    print('DEVICE', json.dumps(device_memory(torch, device)), flush=True)
    history, best_score, stale = [], None, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= 2
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(not frozen)
        fraction = max(0, epoch - 3) / max(1, args.epochs - 3)
        factor = .15 + .85 * (1 + math.cos(math.pi * fraction)) / 2
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        model.train()
        if frozen:
            model.backbone.eval()
        train_total, train_seen = 0., 0
        epoch_started = time.perf_counter()
        for step, (images, breeds, species) in enumerate(train, 1):
            images, breeds, species = images.to(device), breeds.to(device), species.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if amp else nullcontext():
                breed_logits, species_logits = model(images)
                mask = breeds >= 0
                loss_b = breed_loss(breed_logits[mask], breeds[mask]) if bool(mask.any()) else breed_logits.sum() * 0
                loss = loss_b + .5 * species_loss(species_logits, species)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Non-finite train loss epoch={epoch} batch={step}; no retry')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            gpu_checkpoint(pause_seconds=.05)
            train_total += float(loss.detach()) * len(images)
            train_seen += len(images)
            if step == 1 or step % 50 == 0 or step == len(train):
                print('TRAIN', json.dumps({'epoch': epoch, 'batch': step, 'batches': len(train),
                      'loss': train_total / train_seen, **device_memory(torch, device)}), flush=True)
        model.eval()
        support, correct = [0] * len(classes), [0] * len(classes)
        cats_accepted = dogs = false_cat = 0
        val_breed_sum = val_species_sum = 0.
        val_seen = 0
        with torch.inference_mode():
            for step, (images, breeds, species) in enumerate(val, 1):
                images, breeds_gpu, species_gpu = images.to(device), breeds.to(device), species.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if amp else nullcontext():
                    bp, sp = model(images)
                    mask = breeds_gpu >= 0
                    b_loss = breed_loss(bp[mask], breeds_gpu[mask]) if bool(mask.any()) else bp.sum() * 0
                    s_loss = species_loss(sp, species_gpu)
                if not bool(torch.isfinite(b_loss + s_loss)):
                    raise FloatingPointError(f'Non-finite validation loss epoch={epoch} batch={step}; no retry')
                bp, sp = bp.float().cpu(), sp.float().cpu()
                gpu_checkpoint(pause_seconds=.05)
                cat_count = int((species == 1).sum())
                val_breed_sum += float(b_loss) * cat_count
                val_species_sum += float(s_loss) * len(images)
                val_seen += len(images)
                breed_indices, species_indices = bp.argmax(1).tolist(), sp.argmax(1).tolist()
                for truth, species_truth, guess, species_guess in zip(breeds.tolist(), species.tolist(), breed_indices, species_indices):
                    if species_truth:
                        support[truth] += 1
                        correct[truth] += int(guess == truth)
                        cats_accepted += int(species_guess == 1)
                    else:
                        dogs += 1
                        false_cat += int(species_guess == 1)
                if step == 1 or step % 50 == 0 or step == len(val):
                    print(f'VAL epoch={epoch} batch={step}/{len(val)}', flush=True)
        accuracies = [a / n for a, n in zip(correct, support)]
        val_loss = val_breed_sum / sum(support) + .5 * val_species_sum / val_seen
        row = {'epoch': epoch, 'backbone_frozen': frozen, 'train_loss': train_total / train_seen,
               'val_loss': val_loss, 'val_breed_accuracy': sum(correct) / sum(support),
               'val_breed_macro_accuracy': sum(accuracies) / len(classes),
               'val_old12_macro_accuracy': sum(accuracies[:12]) / 12,
               'val_new8_macro_accuracy': sum(accuracies[12:]) / 8,
               'val_per_breed': {name: {'count': n, 'correct': a, 'accuracy': a / n}
                                 for name, n, a in zip(classes, support, correct)},
               'val_cats': sum(support), 'val_cats_accepted': cats_accepted,
               'val_dogs': dogs, 'val_false_cat_dogs': false_cat,
               'learning_rates': [g['lr'] for g in optimizer.param_groups],
               'train_batches': len(train), 'val_batches': len(val),
               'seconds': time.perf_counter() - epoch_started,
               'elapsed_seconds': time.perf_counter() - started, **device_memory(torch, device)}
        history.append(row)
        payload = {'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                   'classes': classes, 'epoch': epoch, 'config': config, 'metrics': row}
        save_checkpoint(out / f'epoch{epoch:02d}.pt', payload, immutable=True)
        save_json(out / 'history.json', history)
        score = (row['val_breed_macro_accuracy'], -val_loss)
        if best_score is None or score > best_score:
            best_score, stale = score, 0
            save_checkpoint(destination, payload)
            save_json(out / 'best.json', row)
        else:
            stale += 1
        print('EPOCH', json.dumps(row), 'BEST', best_score, 'patience', stale, flush=True)
        if epoch >= 10 and stale >= args.patience:
            print('EARLY STOP validation only', flush=True)
            break
    save_json(out / 'complete.json', {'best': json.loads((out / 'best.json').read_text(encoding='utf-8')),
              'epochs_completed': len(history), 'seconds': time.perf_counter() - started,
              'best_weights_sha256': file_sha256(destination), 'no_test_input': True})
    print('TRAINING COMPLETED', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DATA_DIR / 'train-manifest.json')
    parser.add_argument('--classes', type=Path, default=DATA_DIR / 'classes.json')
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.epochs < 3 or args.patience < 1:
        parser.error('--epochs must be >=3 and --patience must be >=1')
    with ModelTask('train-expanded-v3'):
        run(args)


if __name__ == '__main__':
    main()
