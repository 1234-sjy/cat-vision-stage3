"""Regress the fixed current 20-class VLM on the original 120 images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE']='1'
from catvision.common import ROOT, load_image, save_json
from catvision.evaluation import read_verified_input, score_predictions, inference_fingerprint
from catvision.run_control import ModelTask


def main():
    from catvision.vlm import LocalVLM
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    out=ROOT/'artifacts/optimization-v3/vlm-oldset-control'
    if out.exists():raise FileExistsError(out)
    manifest=ROOT/'artifacts/optimization-v2/comparison-v2.json'
    records=json.loads(manifest.read_text(encoding='utf-8'))
    baseline=json.loads((ROOT/'artifacts/evaluation/comparison-v2/vlm/metrics.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    engine=LocalVLM();fingerprint=inference_fingerprint(ROOT,'vlm')
    save_json(out/'config.json',{'images':'exact original comparison-v2 120 images',
        'image_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'active_candidate_count':20,'prompt_version':engine.prompt_version,'prompt':engine.prompt,
        'historical_baseline':{'correct':baseline['correct'],'count':baseline['count'],'accuracy':baseline['accuracy']},
        'purpose':'separate image-set change from the combined candidate/prompt update; no model selection',
        'fingerprint':fingerprint,'fresh_inference':True})
    predictions=[]
    try:
        for i,row in enumerate(records):
            content,sha=read_verified_input(row,args.data_root)
            prediction=engine.predict(load_image(content))
            prediction.update(image_id=row['image_id'],true_breed=row['breed'],input_sha256=sha,
                              run_fingerprint=fingerprint,fresh_run=True)
            predictions.append(prediction);save_json(out/'predictions.json',predictions)
            print(f'CONTROL {i+1}/120 {row["breed"]} -> {prediction.get("breed")} {prediction["inference_seconds"]:.3f}s',flush=True)
    finally:engine.close()
    metrics=score_predictions(records,predictions)
    save_json(out/'metrics.json',metrics)
    print('RESULT',metrics['correct'],metrics['count'],metrics['accuracy'],flush=True)


if __name__=='__main__':
    with ModelTask('vlm-oldset-control-v3'):main()
