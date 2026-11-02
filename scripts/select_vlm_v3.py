"""Predeclared validation-only prompt comparison. No test records accepted."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE']='1'
from catvision.common import ROOT, load_image, save_json
from catvision.evaluation import score_predictions, read_verified_input
from catvision.run_control import ModelTask


def main():
    from catvision.vlm import LocalVLM
    from catvision.vlm_prompts import breed_prompt
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--per-class',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--versions',nargs='+',default=['expanded-baseline-v3','morphology-v3'])
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    source=ROOT/'artifacts/optimization-v3/data/train-manifest.json'
    records=json.loads(source.read_text(encoding='utf-8'))
    val=[r for r in records if r['split']=='val' and r['is_cat']]
    rng=random.Random(20261102)
    chosen=[]
    for breed in sorted({r['breed'] for r in val}):
        group=sorted([r for r in val if r['breed']==breed],key=lambda r:r['relative_path'])
        chosen.extend(rng.sample(group,min(args.per_class,len(group))))
    args.output.mkdir(parents=True)
    save_json(args.output/'selection.json',{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'records':chosen,'selection':'validation only; seed20261102; balanced5/class',
        'candidates':args.versions,'pixels':768*768,
        'choice':'accuracy on this validation subset, then mean latency','test_used':False})
    engine=LocalVLM()
    scores={}
    for version in args.versions:
        engine.prompt=breed_prompt(version); engine.prompt_version=version
        engine.output_format='lines' if 'lines' in version else 'json'
        predictions=[]
        for i,record in enumerate(chosen):
            content,sha=read_verified_input(record,args.data_root)
            prediction=engine.predict(load_image(content))
            prediction.update(image_id=record['image_id'],true_breed=record['breed'],input_sha256=sha)
            predictions.append(prediction)
            save_json(args.output/f'{version}-predictions.json',predictions)
            print(f'{version} {i+1}/{len(chosen)} expected={record["breed"]} predicted={prediction.get("breed")} status={prediction["status"]} seconds={prediction["inference_seconds"]:.3f}',flush=True)
        metric=score_predictions(chosen,predictions);scores[version]=metric
        save_json(args.output/f'{version}-metrics.json',metric)
        save_json(args.output/f'{version}-prompt.json',{'text':engine.prompt,'version':version})
    engine.close()
    selected=max(scores,key=lambda v:(scores[v]['correct'],-scores[v]['mean_inference_seconds']))
    save_json(args.output/'selected.json',{'prompt_version':selected,'scores':scores,'test_used':False})
    print('SELECTED',selected,flush=True)


if __name__=='__main__':
    with ModelTask('select-vlm-v3'): main()
