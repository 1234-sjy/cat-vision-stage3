"""Fresh batched evaluation of exactly the active exported inference configuration."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE']='1'
from catvision.common import ROOT, load_image, save_json
from catvision.evaluation import read_verified_input, inference_fingerprint
from catvision.run_control import ModelTask
from scripts.assess_expanded import score_group, classify_probabilities
from scripts.train_expanded import OLD_CLASSES


def main():
    import torch
    from catvision.breed_transformer import BreedTransformer
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--setval',action='store_true')
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config=json.loads((ROOT/'configs/active-classifier.json').read_text(encoding='utf-8'))
    manifest=ROOT/('artifacts/optimization-v3/data/train-manifest.json' if args.setval else 'artifacts/optimization-v3/data/test-manifest.json')
    records=[r for r in json.loads(manifest.read_text(encoding='utf-8')) if r['split']==('val' if args.setval else 'test')]
    torch.set_float32_matmul_precision('highest')
    engine=BreedTransformer(ROOT/'weights'/config['checkpoint'],config)
    classes=engine.classes;zh=[r['breed_zh'] for r in json.loads((ROOT/'artifacts/optimization-v3/data/classes.json').read_text(encoding='utf-8'))]
    args.output.mkdir(parents=True)
    fingerprint=inference_fingerprint(ROOT,'traditional')
    save_json(args.output/'config.json',{'set':'val' if args.setval else 'test','active_classifier':config,
        'fingerprint':fingerprint,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'count':len(records),'batch':args.batch_size,'precision':'FP32, same exported weights and transforms as web',
        'input_max_side':1600,'fresh_inference':True,'answer_cache':False,'source_counts':dict(Counter(r['source'] for r in records))})
    predictions=[];started=time.perf_counter()
    for start in range(0,len(records),args.batch_size):
        group=records[start:start+args.batch_size];images=[]
        for row in group:
            content,sha=read_verified_input(row,args.data_root)
            images.append(load_image(content,max_side=1600))
        batch=engine.probabilities_batch(images)
        for i,row in enumerate(group):
            answer=classify_probabilities(batch['breed_probs'][i],batch['cat_probability'][i],classes,zh)
            predictions.append({**answer,'image_id':row['image_id'],'true_breed':row['breed'],
                'is_cat':row['is_cat'],'input_sha256':row['sha256'],'group':row['group'],
                'source':row['source'],'evaluation_scope':row['evaluation_scope'],
                'fresh_run':True,'run_fingerprint':fingerprint})
        save_json(args.output/'predictions.json',predictions)
        print('ACTIVE',len(predictions),'/',len(records),flush=True)
    metric={'all':score_group(predictions,classes),'groups':{},'config':config,
        'run_fingerprint':fingerprint,'elapsed_seconds':time.perf_counter()-started,
        'scope':'active original-image classification, including both views; no detector/SAM needed'}
    if not args.setval:
        for scope,names in [('oxford_test',OLD_CLASSES),('expanded_test',classes)]:
            metric['groups'][scope]=score_group([r for r in predictions if r['evaluation_scope']==scope],classes,names)
    else:
        for scope,names in [('old12',classes[:12]),('new8',classes[12:])]:
            metric['groups'][scope]=score_group([r for r in predictions if r['true_breed'] in names],classes,names)
    save_json(args.output/'metrics.json',metric)
    save_json(args.output/'complete.json',{'count':len(predictions),'scope':metric['scope'],'run_fingerprint':fingerprint})
    print('RESULT',metric['all']['correct'],metric['all']['count'],metric['all']['accuracy'],flush=True)


if __name__=='__main__':
    with ModelTask('assess-active-v3'): main()
