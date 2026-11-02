"""Fresh, fingerprinted original-image assessment of the actual 20-class model.

Default: frozen test inputs, FP32, batch 8, workers 0 and a .05s synchronized
pause. --setval explicitly switches to validation-only confidence diagnostics.
Confidence diagnostics never add an unsupported-breed detector or change labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, gpu_checkpoint
from scripts.train_expanded import (DATA_DIR, OLD_CLASSES, NEW_CLASSES, device_memory,
                                    file_sha256, load_classes, validate_records,
                                    verified_image_path)


def classify_probabilities(probabilities: list[float], cat_probability: float,
                           classes: list[str], class_zh: list[str]) -> dict:
    if len(probabilities) != len(classes) or len(classes) != len(class_zh) or len(classes) < 2:
        raise ValueError('Probability/class mapping mismatch')
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise FloatingPointError('Invalid breed probability; assessment stops without retry')
    if not math.isclose(sum(probabilities), 1., abs_tol=1e-4):
        raise ValueError('Breed probabilities do not sum to one')
    if not math.isfinite(cat_probability) or not 0 <= cat_probability <= 1:
        raise FloatingPointError('Invalid species probability')
    order = sorted(range(len(classes)), key=lambda i: probabilities[i], reverse=True)
    first, second = order[:2]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0) / math.log(len(classes))
    accepted = cat_probability >= .5
    return {'status': 'ok' if accepted else 'no_cat', 'has_cat': accepted,
            'breed': classes[first] if accepted else None,
            'breed_zh': class_zh[first] if accepted else None,
            'raw_breed': classes[first], 'breed_score': probabilities[first],
            'cat_probability': cat_probability, 'normalized_entropy': entropy,
            'margin': probabilities[first] - probabilities[second],
            'breed_probabilities': probabilities,
            'top3': [{'breed': classes[index], 'probability': probabilities[index]}
                     for index in order[:3]]}


def score_group(predictions: list[dict], classes: list[str],
                evaluated_classes: list[str] | None = None) -> dict:
    """End-to-end breed scores count a species rejection as a wrong answer."""
    evaluated = list(evaluated_classes if evaluated_classes is not None else classes)
    if not set(evaluated).issubset(classes):
        raise ValueError('Evaluation classes must be output classes')
    cats = [row for row in predictions if row['is_cat']]
    dogs = [row for row in predictions if not row['is_cat']]
    unknown = set(row['true_breed'] for row in cats) - set(classes)
    if unknown:
        raise ValueError(f'Unknown ground truth classes: {sorted(unknown)}')
    columns = list(classes) + ['__no_cat__', '__error__']
    column_index = {name: index for index, name in enumerate(columns)}
    matrix = [[0] * len(columns) for _ in classes]
    tp, fp, fn, support, raw_correct = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in cats:
        truth = row['true_breed']
        guess = row.get('breed') if row['status'] == 'ok' else None
        if guess is not None and guess not in classes:
            raise ValueError('Classifier produced a class outside the checkpoint mapping')
        column = guess if guess is not None else '__no_cat__' if row['status'] == 'no_cat' else '__error__'
        matrix[classes.index(truth)][column_index[column]] += 1
        support[truth] += 1
        raw_correct[truth] += int(row.get('raw_breed') == truth)
        if guess == truth:
            tp[truth] += 1
        else:
            fn[truth] += 1
            if guess is not None:
                fp[guess] += 1
    per_breed = {}
    for name in classes:
        precision = tp[name] / (tp[name] + fp[name]) if tp[name] + fp[name] else 0.
        recall = tp[name] / support[name] if support[name] else 0.
        per_breed[name] = {'count': support[name], 'correct': tp[name],
                           'accuracy': recall, 'precision': precision, 'recall': recall,
                           'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                           'raw_head_correct': raw_correct[name],
                           'raw_head_accuracy': raw_correct[name] / support[name] if support[name] else 0.}
    count = len(cats)
    false_cat = sum(row['status'] == 'ok' for row in dogs)
    return {'count': count, 'correct': sum(tp.values()),
            'accuracy': sum(tp.values()) / count if count else None,
            'macro_accuracy': sum(per_breed[name]['accuracy'] for name in evaluated) / len(evaluated) if count else None,
            'macro_f1': sum(per_breed[name]['f1'] for name in evaluated) / len(evaluated) if count else None,
            'macro_averaged_breeds': evaluated,
            'raw_head_accuracy': sum(raw_correct.values()) / count if count else None,
            'per_breed': per_breed, 'status_counts': dict(Counter(row['status'] for row in predictions)),
            'confusion_rows': list(classes), 'confusion_columns': columns, 'confusion_matrix': matrix,
            'dog_count': len(dogs), 'dog_false_cat': false_cat,
            'dog_false_cat_rate': false_cat / len(dogs) if dogs else None,
            'species_cat_acceptance': sum(row['status'] == 'ok' for row in cats) / count if count else None,
            'scope': 'original-image classifier only; no detector, segmentation, crop or fusion'}


def confidence_diagnostics(predictions: list[dict]) -> dict:
    """Validation-only descriptive coverage/risk; no automatic threshold choice."""
    cats = [row for row in predictions if row['is_cat']]
    rows = []
    for measure, thresholds, at_least in [
        ('breed_score', [.4, .5, .6, .7, .8, .9, .95], True),
        ('normalized_entropy', [.1, .2, .3, .4, .5, .6, .75], False),
        ('margin', [.05, .1, .2, .3, .4, .5], True)]:
        for threshold in thresholds:
            accepted = [row for row in cats if row['status'] == 'ok' and
                        (row[measure] >= threshold if at_least else row[measure] <= threshold)]
            correct = sum(row['breed'] == row['true_breed'] for row in accepted)
            rows.append({'measure': measure, 'threshold': threshold,
                         'operator': '>=' if at_least else '<=', 'accepted': len(accepted),
                         'known_cat_count': len(cats),
                         'coverage': len(accepted) / len(cats) if cats else None,
                         'accuracy_when_covered': correct / len(accepted) if accepted else None,
                         'error_when_covered': 1 - correct / len(accepted) if accepted else None})
    return {'set': 'validation', 'automatic_threshold_selection': False,
            'inference_decision_changed': False,
            'limitation': 'Known-class confidence/ambiguity only; no unseen-breed detection claim.',
            'coverage_risk': rows}


def run(args) -> None:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import ImageOps
    from catvision.breed_transformer import BreedSwin, eval_transform

    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(2)
    # Explicit FP32 assessment: no BF16/autocast and no TF32 approximation.
    torch.set_float32_matmul_precision('highest')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; stopping without fallback')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    classes, class_zh = load_classes(args.classes)
    selected_set = 'val' if args.setval else 'test'
    manifest = args.manifest or DATA_DIR / ('train-manifest.json' if args.setval else 'test-manifest.json')
    raw_records = json.loads(manifest.read_text(encoding='utf-8'))
    validate_records(raw_records, classes, {'train', 'val'} if args.setval else {'test'})
    records = [row for row in raw_records if row['split'] == selected_set]
    if not records:
        raise ValueError(f'Empty {selected_set} set')
    if not args.setval and any(row.get('evaluation_scope') not in {'oxford_test', 'expanded_test'} for row in records):
        raise ValueError('Test manifest requires frozen oxford_test/expanded_test scope labels')
    weights_sha = file_sha256(args.weights)
    manifest_sha = file_sha256(manifest)
    classes_sha = file_sha256(args.classes)
    payload = torch.load(args.weights, map_location='cpu', weights_only=True)
    if payload['classes'] != classes:
        raise ValueError('Checkpoint does not have the frozen 20-class mapping')
    training_config = payload.get('config', {})
    if training_config.get('classes_sha256', classes_sha) != classes_sha:
        raise ValueError('Class metadata changed since training')
    if args.setval and training_config.get('train_manifest_sha256', manifest_sha) != manifest_sha:
        raise ValueError('Validation manifest changed since training')
    # Reserve a genuinely new output directory; partial and complete old runs stay intact.
    args.output.mkdir(parents=True, exist_ok=False)
    config = {'set': selected_set, 'weights': str(args.weights.resolve()),
              'weights_sha256': weights_sha, 'checkpoint_epoch': payload.get('epoch'),
              'manifest': str(manifest.resolve()), 'manifest_sha256': manifest_sha,
              'classes_file': str(args.classes.resolve()), 'classes_sha256': classes_sha,
              'classes': classes, 'classes_zh': class_zh, 'data_root': str(args.data_root.resolve()),
              'count': len(records), 'source_counts': dict(Counter(row['source'] for row in records)),
              'evaluation_scope_counts': dict(Counter(row['evaluation_scope'] for row in records)),
              'precision': 'FP32; BF16/FP16/TF32 disabled', 'device': str(device),
              'batch': 8, 'workers': 0, 'cpu_threads': 2, 'pause_after_sync_seconds': .05,
              'tta': args.tta, 'cat_threshold': .5, 'breed_ambiguity_rejection': False,
              'threshold_selected_on_test': False, 'fresh_inference': True,
              'automatic_retry': False, 'automatic_resume': False,
              'torch_version': str(torch.__version__)}
    save_json(args.output / 'config.json', config)
    save_json(args.output / f'{selected_set}-manifest.json', records)
    print('CONFIG', json.dumps(config), flush=True)
    transform = eval_transform()

    class FrozenDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            row = records[index]
            image = load_image(verified_image_path(row, args.data_root))
            original = transform(image)
            flipped = transform(ImageOps.mirror(image)) if args.tta else original
            return original, flipped, index

    loader = DataLoader(FrozenDataset(), batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    model = BreedSwin(classes=classes)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.float().to(device).eval()
    del payload
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (original, flipped, indices) in enumerate(loader, 1):
            breed_sums = species_sums = None
            for images in ([original, flipped] if args.tta else [original]):
                bp, sp = model(images.to(device))
                bp, sp = bp.softmax(-1).cpu(), sp.softmax(-1).cpu()
                gpu_checkpoint(pause_seconds=.05)
                breed_sums = bp if breed_sums is None else breed_sums + bp
                species_sums = sp if species_sums is None else species_sums + sp
            view_count = 2 if args.tta else 1
            bp, sp = breed_sums / view_count, species_sums / view_count
            for index, row_breed, row_species in zip(indices.tolist(), bp.tolist(), sp.tolist()):
                row = records[index]
                answer = classify_probabilities(row_breed, row_species[1], classes, class_zh)
                predictions.append({**answer, 'image_id': row['image_id'],
                                    'relative_path': row['relative_path'],
                                    'true_breed': row['breed'], 'is_cat': row['is_cat'],
                                    'input_sha256': row['sha256'], 'group': row['group'],
                                    'source': row['source'], 'evaluation_scope': row['evaluation_scope']})
            if step == 1 or step % 25 == 0 or step == len(loader):
                save_json(args.output / 'predictions.json', predictions)
                print('ASSESS', json.dumps({'set': selected_set, 'batch': step, 'batches': len(loader),
                      'predictions': len(predictions), 'count': len(records),
                      **device_memory(torch, device)}), flush=True)
    if len(predictions) != len(records):
        raise RuntimeError('Incomplete fresh prediction set')
    if file_sha256(args.weights) != weights_sha or file_sha256(manifest) != manifest_sha:
        raise ValueError('Weights or manifest changed during assessment')
    save_json(args.output / 'predictions.json', predictions)
    metrics = {'all': score_group(predictions, classes), 'groups': {},
               'weights_sha256': weights_sha, 'manifest_sha256': manifest_sha,
               'classes_sha256': classes_sha, 'set': selected_set,
               'elapsed_seconds': time.perf_counter() - started, 'config': config}
    if not args.setval:
        for scope, evaluated in [('oxford_test', OLD_CLASSES), ('expanded_test', classes)]:
            selected = [row for row in predictions if row['evaluation_scope'] == scope]
            metrics['groups'][scope] = score_group(selected, classes, evaluated)
    else:
        for title, names in [('old12', OLD_CLASSES), ('new8', NEW_CLASSES)]:
            selected = [row for row in predictions if row['true_breed'] in names]
            metrics['groups'][title] = score_group(selected, classes, names)
        save_json(args.output / 'validation-confidence.json', confidence_diagnostics(predictions))
    # Cohort-specific class counts and statuses stay visible alongside aggregate metrics.
    if not metrics['all']['dog_count']:
        metrics['dog_scope_note'] = 'Frozen selected set contains no dogs; dog rejection performance is not estimated.'
    metrics['known_class_scope_note'] = 'Measures named 20-class classification; no unseen-breed rejection validation.'
    save_json(args.output / 'metrics.json', metrics)
    save_json(args.output / 'complete.json', {'count': len(predictions), 'set': selected_set,
              'weights_sha256': weights_sha, 'manifest_sha256': manifest_sha,
              'predictions_sha256': file_sha256(args.output / 'predictions.json'),
              'metrics_sha256': file_sha256(args.output / 'metrics.json'),
              'elapsed_seconds': time.perf_counter() - started})
    print('RESULT', json.dumps({'set': selected_set, 'all': {key: metrics['all'][key] for key in
          ['count', 'correct', 'accuracy', 'macro_accuracy', 'macro_f1', 'status_counts']},
          'groups': {name: {key: group[key] for key in ['count', 'accuracy', 'macro_f1']}
                     for name, group in metrics['groups'].items()}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/breed-swin-v3-generic.pt')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--classes', type=Path, default=DATA_DIR / 'classes.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--setval', action='store_true', help='Validation only, including confidence/coverage diagnostics')
    parser.add_argument('--tta', action='store_true', help='Average original and horizontal-flip probabilities')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    with ModelTask('assess-expanded-v3-val' if args.setval else 'assess-expanded-v3-test'):
        run(args)


if __name__ == '__main__':
    main()
