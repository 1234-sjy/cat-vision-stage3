"""Evaluate a validation-selected adapter on a frozen cohort, without retries."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE'] = '1'

from catvision.common import ROOT, BREEDS, load_image, save_json
from catvision.evaluation import read_verified_input, score_predictions, inference_fingerprint, validate_comparison
from catvision.run_control import ModelTask


def grouped_metrics(records, predictions):
    by_id = {p['image_id']: p for p in predictions}
    groups = {'all': records,
        'original_12_breeds': [r for r in records if r['breed'] in list(BREEDS)[:12]],
        'added_8_breeds': [r for r in records if r['breed'] in list(BREEDS)[12:]],
        'oxford_source': [r for r in records if r['source'] == 'oxford_iiit_pet_official'],
        'expanded_source': [r for r in records if r['source'] != 'oxford_iiit_pet_official']}
    return {name: score_predictions(rows, [by_id[r['image_id']] for r in rows])
            for name, rows in groups.items() if rows}


def main(args):
    import torch
    from catvision.vlm import LocalVLM
    from catvision.vlm_training import collate_processor
    from scripts.train_vlm_lora import QuietTelemetry
    from catvision.vlm_format import parse_breed_lines

    data = ROOT/'artifacts/optimization-v4/data'
    manifest = data/f'{args.cohort}.json'
    freeze = json.loads((data/'freeze.json').read_text(encoding='utf-8'))
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != freeze['files'][manifest.name]:
        raise ValueError('Frozen assessment manifest changed')
    records = json.loads(manifest.read_text(encoding='utf-8'))
    if args.output.exists():
        raise FileExistsError('Existing assessment retained; no automatic overwrite/retry')
    selection_path = ROOT/'artifacts/optimization-v4/final-selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    settings_path = ROOT/'configs/active-vlm.json'
    active = json.loads(settings_path.read_text(encoding='utf-8'))
    if selection['active_config_sha256'] != hashlib.sha256(settings_path.read_bytes()).hexdigest():
        raise ValueError('The validation-selected configuration changed before evaluation')
    args.output.mkdir(parents=True)
    engine = None
    predictions = []
    with nullcontext(QuietTelemetry()) as telemetry:
        try:
            telemetry.set_context('model_loading')
            engine = LocalVLM()
            fingerprint = inference_fingerprint(ROOT, 'vlm')
            save_json(args.output/'config.json', {
                'created_utc': datetime.now(timezone.utc).isoformat(), 'cohort': args.cohort,
                'count': len(records), 'manifest_sha256': actual, 'active_vlm': active,
                'fingerprint': fingerprint, 'selection': selection,
                'batch_size': args.batch, 'input_max_side': 1600,
                'fresh_inference': True, 'answer_cache': False,
                'previously_evaluated_cohort': args.cohort == 'shared-regression',
                'used_for_training_or_selection': False})
            started = time.perf_counter()
            for offset in range(0, len(records), args.batch):
                batch = records[offset:offset+args.batch]
                telemetry.set_context('assessment', batch=offset//args.batch+1)
                samples = []
                for row in batch:
                    content, _ = read_verified_input(row, args.data_root)
                    samples.append({'image': load_image(content, max_side=1600), 'record': row})
                torch.cuda.synchronize()
                batch_started = time.perf_counter()
                inputs = collate_processor(engine.processor, samples, engine.prompt, training=False)
                inputs = inputs.to(engine.model.device)
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    generated = engine.model.generate(**inputs, max_new_tokens=engine.max_new_tokens,
                                                      do_sample=False, use_cache=True)
                answers = generated[:, inputs['input_ids'].shape[1]:]
                texts = engine.processor.batch_decode(answers, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-batch_started
                for row, text in zip(batch, texts):
                    result = parse_breed_lines(text)
                    result.update(image_id=row['image_id'], true_breed=row['breed'],
                        input_sha256=row['sha256'], source=row['source'],
                        run_fingerprint=fingerprint, fresh_run=True,
                        batch_inference_seconds=elapsed, batch_size=len(batch),
                        prompt_version=engine.prompt_version, adapter=engine.adapter)
                    predictions.append(result)
                save_json(args.output/'predictions.json', predictions)
                print(f'ASSESS {len(predictions)}/{len(records)} batch_seconds={elapsed:.3f}', flush=True)
                del inputs, generated, answers, samples
            validate_comparison(records, predictions)
            metrics = grouped_metrics(records, predictions)
            save_json(args.output/'metrics.json', metrics)
            save_json(args.output/'complete.json', {'count': len(records),
                'seconds': time.perf_counter()-started, 'fresh_inference': True,
                'correct': metrics['all']['correct'], 'accuracy': metrics['all']['accuracy']})
            print('RESULT', json.dumps({k: {'correct': v['correct'], 'count': v['count'],
                'accuracy': v['accuracy']} for k, v in metrics.items()}), flush=True)
        finally:
            if engine is not None:
                engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cohort', choices=['fresh-test', 'shared-regression'], default='fresh-test')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    with ModelTask(f'assess-vlm-v4-{args.cohort}'):
        main(args)
