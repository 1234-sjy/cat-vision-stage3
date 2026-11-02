"""Freeze or evaluate a shared 120-image, 20-breed test; fresh two-route runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.evaluation import read_verified_input, score_predictions, inference_fingerprint, validate_comparison
from catvision.run_control import ModelTask, GpuReservation

OUT=ROOT/'artifacts/optimization-v3/shared-test'


def freeze():
    manifest=ROOT/'artifacts/optimization-v3/data/test-manifest.json'
    records=json.loads(manifest.read_text(encoding='utf-8'))
    names=[r['breed'] for r in json.loads((ROOT/'artifacts/optimization-v3/data/classes.json').read_text(encoding='utf-8'))]
    rng=random.Random(20261102)
    chosen=[]
    for name in names:
        group=sorted([r for r in records if r['breed']==name],key=lambda r:r['image_id'])
        original=[r for r in group if r['evaluation_scope']=='oxford_test']
        expanded=[r for r in group if r['evaluation_scope']=='expanded_test']
        if original:
            chosen.extend(rng.sample(original,3));chosen.extend(rng.sample(expanded,3))
        else:
            chosen.extend(rng.sample(expanded,6))
    if len(chosen)!=120: raise ValueError('Expected120 shared images')
    if OUT.exists(): raise FileExistsError(OUT)
    save_json(OUT/'manifest.json',chosen)
    save_json(OUT/'freeze.json',{'seed':20261102,'count':120,'classes':names,'per_class':6,
        'selection':'old12:3Oxford+3expanded/class; new8:6expanded/class; no performance-based selection',
        'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),
        'frozen_before_test':True})
    print('FROZEN120 shared20-class test inputs',flush=True)


def evaluate(data_root):
    os.environ['CATVISION_FAST_INFERENCE']='1'
    from catvision.server import LocalComparison
    records=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if (OUT/'traditional-predictions.json').exists(): raise FileExistsError('Existing assessment retained')
    frozen=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
    if hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()!=frozen['manifest_sha256']:
        raise ValueError('Shared test manifest changed')
    engine=LocalComparison(keep_loaded=True)
    predictions={route:[] for route in ['traditional','vlm']}
    with GpuReservation():
        engine.preload()
        with ModelTask('shared20-test-v3'):
            identities={route:inference_fingerprint(ROOT,route) for route in predictions}
            save_json(OUT/'run-config.json',{'run_fingerprints':identities,'active_classifier':json.loads((ROOT/'configs/active-classifier.json').read_text(encoding='utf-8')),
                'active_vlm':json.loads((ROOT/'configs/active-vlm.json').read_text(encoding='utf-8')),
                'fresh_run':True,'answer_cache':False,'manifest_sha256':frozen['manifest_sha256']})
            try:
                for i,row in enumerate(records):
                    content,sha=read_verified_input(row,data_root)
                    folder=OUT/'images'/f'{i:03d}'
                    folder.mkdir(parents=True,exist_ok=False)
                    result=engine._run(load_image(content,max_side=1600),folder,lambda text:None)
                    for route in predictions:
                        prediction=result[route]
                        prediction.update(image_id=row['image_id'],true_breed=row['breed'],input_sha256=sha,
                            run_fingerprint=identities[route],fresh_run=True,source=row['source'])
                        predictions[route].append(prediction)
                        save_json(OUT/f'{route}-predictions.json',predictions[route])
                    print(f'SHARED {i+1}/120 {row["breed"]} A={result["traditional"].get("breed")} B={result["vlm"].get("breed")} seconds={result["total_seconds"]:.3f}',flush=True)
            finally:
                engine.close()
            validate_comparison(records,*predictions.values())
            for route,rows in predictions.items():
                metric=score_predictions(records,rows)
                save_json(OUT/f'{route}-metrics.json',metric)
                print('METRICS',route,metric['correct'],metric['count'],metric['accuracy'],flush=True)
            save_json(OUT/'complete.json',{'count':120,'routes':list(predictions),'fresh_run':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--data-root',type=Path)
    args=parser.parse_args()
    if args.freeze: freeze()
    elif args.data_root: evaluate(args.data_root)
    else: parser.error('--data-root required for inference')
