"""Frozen classifier-only test: all official cat test images plus held-out dogs.

This measures the original-image branch, not detector/SAM/fusion pipeline accuracy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT,load_image,save_json
from catvision.evaluation import score_predictions,read_verified_input
from catvision.run_control import ModelTask,gpu_checkpoint,set_low_load_threads


def main():
    import torch
    from PIL import ImageOps
    from catvision.breed_transformer import BreedTransformer,CLASSES,decision
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args();set_low_load_threads()
    out=ROOT/'artifacts/optimization-v2/full-original-test'
    if (out/'metrics.json').exists():raise FileExistsError('Existing test retained; no automatic rerun')
    config=json.loads((ROOT/'configs/active-classifier.json').read_text())
    records=[r for r in json.loads((ROOT/'artifacts/data/manifest.json').read_text()) if r['split']=='test']
    groups=defaultdict(list)
    for line in (args.data_root/'annotations/test.txt').read_text().splitlines():
        fields=line.split()
        if fields and not fields[0].startswith('#') and int(fields[2])==2:
            groups[fields[0].rsplit('_',1)[0]].append(fields[0])
    rng=random.Random(20261102)
    dogs=[]
    for breed in sorted(groups):
        for key in rng.sample(sorted(groups[breed]),4):
            path=args.data_root/f'images/{key}.jpg'
            dogs.append({'image_id':key,'relative_path':f'images/{key}.jpg','breed':'negative_dog',
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':'test'})
    save_json(out/'test-manifest.json',records+dogs)
    model=BreedTransformer(ROOT/'weights'/config['checkpoint'],config)
    predictions=[]
    with torch.inference_mode():
        for first in range(0,len(records)+len(dogs),2):
            group=(records+dogs)[first:first+2]
            images=[load_image(read_verified_input(r,args.data_root)[0]) for r in group]
            sums_b=[];sums_s=[]
            for flip in ([False,True] if config['tta'] else [False]):
                x=torch.stack([model.transform(ImageOps.mirror(im) if flip else im) for im in images]).to(model.device)
                b,s=model.model(x);sums_b.append(b.softmax(-1).cpu());sums_s.append(s.softmax(-1).cpu())
                gpu_checkpoint(pause_seconds=0.2)
            bp=torch.stack(sums_b).mean(0);sp=torch.stack(sums_s).mean(0)
            for j,r in enumerate(group):
                probs={'breed_probs':bp[j].tolist(),'cat_probability':float(sp[j,1])}
                answer=decision(probs,model.classes,config['cat_threshold'])
                predictions.append(dict(answer,image_id=r['image_id'],true_breed=r['breed'],input_sha256=r['sha256']))
            if first%40==0 or first+2>=len(records)+len(dogs):
                save_json(out/'predictions.json',predictions)
                print(f'ORIGINAL TEST {len(predictions)}/{len(records)+len(dogs)}',flush=True)
    cats=[p for p in predictions if p['true_breed']!='negative_dog'];negative=[p for p in predictions if p['true_breed']=='negative_dog']
    metrics=score_predictions(records,cats)
    metrics.update(scope='classifier original-image branch only; excludes detector/SAM/fusion',
                   dog_test_count=len(negative),dog_false_cat=sum(p['status']=='ok' for p in negative),
                   model_config=config)
    save_json(out/'metrics.json',metrics)
    print('RESULT',metrics['correct'],metrics['count'],metrics['accuracy'],'DOG FP',metrics['dog_false_cat'],len(negative),flush=True)


if __name__=='__main__':
    with ModelTask('original-classifier-test'):
        main()
