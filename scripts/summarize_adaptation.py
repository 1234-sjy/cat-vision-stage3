"""Compile final same-input comparisons from retained, verifiable predictions."""
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, BREEDS, save_json
from catvision.evaluation import validate_comparison
from scripts.assess_vlm_adapted import grouped_metrics

BASE = ROOT/'artifacts/optimization-v4'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = BASE/'comparison-summary.json'
    if target.exists():
        raise FileExistsError(target)
    selection = read(BASE/'final-selection.json')
    active = read(ROOT/'configs/active-vlm.json')
    if selection['active_config_sha256'] != sha(ROOT/'configs/active-vlm.json'):
        raise ValueError('Active visual model changed after final selection')
    source = ROOT/'artifacts/optimization-v3/frozen-test'
    if read(source/'config.json')['active_classifier'] != read(ROOT/'configs/active-classifier.json'):
        raise ValueError('Retained classifier results use a different configuration')
    all_predictions = {r['image_id']: r for r in read(source/'predictions.json')}
    fresh_truth = read(BASE/'data/fresh-test.json')
    fresh_vlm = read(BASE/'fresh-test/predictions.json')
    fresh_traditional = [all_predictions[row['image_id']] for row in fresh_truth]
    validate_comparison(fresh_truth, fresh_traditional, fresh_vlm)
    a_metrics = grouped_metrics(fresh_truth, fresh_traditional)
    b_metrics = grouped_metrics(fresh_truth, fresh_vlm)
    save_json(BASE/'fresh-test/traditional-metrics.json', a_metrics)
    regression_truth = read(BASE/'data/shared-regression.json')
    regression_b = read(BASE/'shared-regression/predictions.json')
    regression_a_source = ROOT/'artifacts/optimization-v3/shared-test/traditional-predictions.json'
    regression_config = read(regression_a_source.parent/'run-config.json')
    if regression_config['active_classifier'] != read(ROOT/'configs/active-classifier.json'):
        raise ValueError('Retained regression classifier configuration changed')
    regression_a = read(regression_a_source)
    validate_comparison(regression_truth, regression_a, regression_b)
    regression = {'traditional': grouped_metrics(regression_truth, regression_a),
                  'vlm': grouped_metrics(regression_truth, regression_b)}
    training = read(ROOT/active['selection_source'])
    summary = {'adapter_sha256': active['adapter_sha256'],
        'active_vlm': active, 'selection': selection,
        'traditional_full_test': read(source/'metrics.json'),
        'fresh_matched': {'count': len(fresh_truth), 'traditional': a_metrics, 'vlm': b_metrics,
            'vlm_fresh_inference': True, 'traditional_reuses_retained_frozen_predictions': True,
            'traditional_source_sha256': sha(source/'predictions.json'),
            'input_manifest_sha256': sha(BASE/'data/fresh-test.json')},
        'regression_120': {**regression, 'previously_evaluated_images': True,
            'before_vlm': read(ROOT/'artifacts/optimization-v3/shared-test/vlm-metrics.json'),
            'traditional_source_sha256': sha(regression_a_source)},
        'training': {key: training[key] for key in ('epochs_completed', 'global_step', 'best_epoch',
            'best_validation', 'seconds')},
        'hyperparameters': {key: training['config'][key] for key in (
            'training_records', 'validation_records', 'learning_rate', 'weight_decay',
            'warmup_ratio', 'lr_schedule', 'gradient_clip_norm', 'effective_batch', 'microbatch',
            'gradient_accumulation', 'input_image_pixels_max', 'max_new_tokens', 'precision', 'seed')},
        'lora': {key: training['config']['lora'][key] for key in (
            'r', 'alpha', 'dropout', 'target_modules', 'trainable_numel', 'total_numel')},
        'data_counts': {'cat_train': 4157, 'dog_train': 999, 'cat_validation': 925,
            'dog_validation': 500, 'original_cat_test': 1872, 'fresh_vlm_test': len(fresh_truth)},
        'no_test_selection': True}
    save_json(target, summary)
    with (BASE/'fresh-test/per-class-comparison.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['品种', '测试图片数', '传统分类器正确率', '视觉大模型正确率'])
        for name in BREEDS:
            a = a_metrics['all']['per_class'][name]
            b = b_metrics['all']['per_class'][name]
            writer.writerow([BREEDS[name], a['support'], f"{a['recall']:.2%}", f"{b['recall']:.2%}"])
    print('FINAL SAME-INPUT RESULTS', json.dumps({route: metrics['all']['accuracy']
          for route, metrics in [('traditional', a_metrics), ('vlm', b_metrics)]}), flush=True)


if __name__ == '__main__':
    main()
