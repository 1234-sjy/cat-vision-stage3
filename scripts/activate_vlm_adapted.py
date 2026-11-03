"""Activate a validation-selected adapter before opening any fresh test input."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.vlm_prompts import VLM_CLASS_PROMPT_VERSION


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.training_dir/'complete.json').read_text(encoding='utf-8'))
    if complete.get('state') != 'complete' or not complete.get('no_test_input'):
        raise ValueError('Training must finish using train/validation inputs only')
    adapter = Path(complete['adapter_output']).resolve()
    if not adapter.is_relative_to(ROOT):
        raise ValueError('Deploy adapter inside project before activation')
    if sha(adapter/'adapter_model.safetensors') != complete['best_adapter_sha256']:
        raise ValueError('Published adapter differs from the validation-selected checkpoint')
    selection_path = ROOT/'artifacts/optimization-v4/final-selection.json'
    if selection_path.exists():
        raise FileExistsError('Final selection already frozen; retain existing assessment')
    metrics = complete['best_validation']
    config = complete['config']
    settings = {'model': 'Qwen/Qwen3-VL-4B-Instruct',
        'adapter': adapter.relative_to(ROOT).as_posix(),
        'adapter_sha256': complete['best_adapter_sha256'],
        'prompt_version': VLM_CLASS_PROMPT_VERSION, 'output_format': 'lines',
        'max_pixels': config['input_image_pixels_max'], 'max_new_tokens': 24,
        'accelerate_patch': True, 'runtime_precision': 'BF16 AMP',
        'validation_count': metrics['cat_total'], 'validation_correct': metrics['cat_correct'],
        'validation_dog_count': metrics['dog']['total'],
        'validation_dog_false_cat_rate': metrics['dog']['false_cat_rate'],
        'selection_source': (args.training_dir.resolve()/'complete.json').relative_to(ROOT).as_posix(),
        'test_used': False}
    active_path = ROOT/'configs/active-vlm.json'
    backup = ROOT/'artifacts/optimization-v4/active-vlm-before-lora-v4.json'
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(active_path, backup)
    save_json(active_path, settings)
    save_json(selection_path, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'active_config_sha256': sha(active_path), 'adapter_sha256': complete['best_adapter_sha256'],
        'training_complete_sha256': sha(args.training_dir/'complete.json'),
        'best_epoch': complete['best_epoch'], 'validation': metrics,
        'selection': 'Fixed validation cat accuracy; dog false-cat rate breaks ties',
        'no_test_input_for_selection': True})
    print('ACTIVATED', complete['best_epoch'], metrics['cat_correct'], metrics['cat_total'], flush=True)
