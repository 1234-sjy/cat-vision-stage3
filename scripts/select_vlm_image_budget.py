"""Compare two inference image budgets on validation only, after training."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE'] = '1'
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask


def main(args):
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    from catvision.vlm import LocalVLM
    from catvision.vlm_prompts import VLM_CLASS_PROMPT, VLM_CLASS_PROMPT_VERSION
    from catvision.vlm_training import (VlmDataset, identity_collate, load_classes,
        verified_image_paths, validate_records, verify_frozen_manifest, file_sha256)
    from scripts.train_vlm_lora import evaluate, QuietTelemetry

    complete = json.loads((args.training_dir/'complete.json').read_text(encoding='utf-8'))
    if complete['state'] != 'complete':
        raise ValueError('Training must be complete before sequential validation')
    adapter = Path(complete['adapter_output'])
    if file_sha256(adapter/'adapter_model.safetensors') != complete['best_adapter_sha256']:
        raise ValueError('Adapter changed since validation selection')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    data = ROOT/'artifacts/optimization-v4/data'
    freeze = json.loads((data/'freeze.json').read_text(encoding='utf-8'))
    manifest = data/'val-selection.json'
    manifest_sha = verify_frozen_manifest(manifest, freeze)
    classes = load_classes(data/'classes.json')
    records = json.loads(manifest.read_text(encoding='utf-8'))
    validate_records(records, classes, 'val')
    dataset = VlmDataset(records, verified_image_paths(records, args.data_root))
    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=identity_collate)
    engine = None
    candidates = []
    try:
        engine = LocalVLM(prompt=VLM_CLASS_PROMPT, prompt_version=VLM_CLASS_PROMPT_VERSION,
            max_pixels=512*512, max_new_tokens=24, output_format='lines', adapter_dir=adapter)
        for pixels in (512*512, 768*768):
            engine.processor = AutoProcessor.from_pretrained(ROOT/'weights/qwen3-vl-4b',
                local_files_only=True, min_pixels=128*128, max_pixels=pixels)
            result = evaluate(engine, loader, classes, QuietTelemetry(), epoch=complete['best_epoch'],
                prompt=VLM_CLASS_PROMPT, prompt_version=VLM_CLASS_PROMPT_VERSION,
                provenance={'adapter_sha256': complete['best_adapter_sha256'],
                    'manifest_sha256': manifest_sha, 'max_pixels': pixels})
            save_json(args.output/f'{pixels}-validation.json', result)
            candidates.append({'max_pixels': pixels, 'metrics': result['metrics'],
                               'seconds': result['seconds']})
            print('IMAGE BUDGET', pixels, result['metrics']['cat_correct'],
                  result['metrics']['cat_total'], result['seconds'], flush=True)
    finally:
        if engine is not None:
            engine.close()
    best = max(candidates, key=lambda row: (row['metrics']['cat_accuracy'],
        -row['metrics']['dog']['false_cat_rate'], -row['max_pixels']))
    save_json(args.output/'selection.json', {'chosen_max_pixels': best['max_pixels'],
        'validation': best['metrics'], 'candidates': candidates,
        'adapter_sha256': complete['best_adapter_sha256'], 'manifest_sha256': manifest_sha,
        'prompt_version': VLM_CLASS_PROMPT_VERSION, 'test_used': False,
        'selection': 'validation cat accuracy, dog false-cat tie break, then smaller pixel budget'})
    print('CHOSEN', best['max_pixels'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with ModelTask('select-vlm-image-budget-v4'):
        main(args)
