"""One validation sweep selects fusion/presence policy and exports portable weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, gpu_checkpoint, set_low_load_threads


def main():
    import torch
    from PIL import ImageOps
    from catvision.breed_transformer import BreedTransformer,CLASSES,combine_views,decision
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    set_low_load_threads()
    out=ROOT/'artifacts/optimization-v2'
    if not (out/'training/complete.json').exists():
        raise RuntimeError('Training must finish before validation selection')
    if (out/'selection.json').exists():
        raise FileExistsError('Selection already completed; no automatic rerun')
    payload=torch.load(ROOT/'weights/breed-swin-v2.pt',map_location='cpu',weights_only=True)
    # Storage-only FP16: inference converts into the FP32 model. Keep original FP32 weights.
    payload['state_dict']={k:v.half() if v.is_floating_point() else v for k,v in payload['state_dict'].items()}
    payload['storage_dtype']='float16; inference float32'
    portable=ROOT/'weights/breed-swin-v2-fp16.pt'
    if portable.exists(): raise FileExistsError('Portable checkpoint already exists')
    torch.save(payload,portable)
    model=BreedTransformer(portable,{'tta':True,'subject_weight':0.5,'cat_threshold':0.5})
    records=[r for r in json.loads((out/'train-manifest.json').read_text()) if r['split']=='val']
    views={r['image_id']:r for r in map(json.loads,(ROOT/'artifacts/views/records.jsonl').read_text().splitlines())}
    @torch.inference_mode()
    def predict_pair(image):
        x=torch.stack([model.transform(image),model.transform(ImageOps.mirror(image))]).to(model.device)
        bp,sp=model.model(x)
        bp=bp.softmax(-1).cpu();sp=sp.softmax(-1).cpu()
        gpu_checkpoint(pause_seconds=0.2)
        return {'single':{'breed_probs':bp[0].tolist(),'cat_probability':float(sp[0,1])},
                'flip':{'breed_probs':bp.mean(0).tolist(),'cat_probability':float(sp.mean(0)[1])}}
    rows=[]
    for i,r in enumerate(records):
        image=load_image(args.data_root/r['relative_path'])
        row={'image_id':r['image_id'],'breed':r['breed'],'is_cat':r['is_cat'],
             'original':predict_pair(image),'detected':False}
        if r['is_cat'] and views[r['image_id']]['status']=='ok':
            row['subject']=predict_pair(image.crop(views[r['image_id']]['crop_box']))
            row['detected']=True
        rows.append(row)
        if (i+1)%20==0 or i+1==len(records):
            save_json(out/'swin-validation-probabilities.json',rows)
            print(f'VALIDATION {i+1}/{len(records)}',flush=True)
    trials=[]
    for tta in [False,True]:
        key='flip' if tta else 'single'
        dog_scores=sorted([r['original'][key]['cat_probability'] for r in rows if not r['is_cat']],reverse=True)
        allowed=math.floor(len(dog_scores)*0.02)
        threshold=max(0.5,math.nextafter(dog_scores[allowed],math.inf))
        for weight in [0.0,0.25,0.5,0.75,1.0]:
            correct=accepted=dog_fp=0
            for r in rows:
                probs=combine_views(r['original'][key],r.get('subject',{}).get(key),weight)
                answer=decision(probs,CLASSES,threshold)
                if r['is_cat']:
                    accepted+=answer['status']=='ok'
                    correct+=answer['status']=='ok' and answer['breed']==r['breed']
                else: dog_fp+=answer['status']=='ok'
            trials.append({'tta':tta,'subject_weight':weight,'cat_threshold':threshold,
                           'correct':correct,'cats':238,'accepted_cats':accepted,
                           'false_cat_dogs':dog_fp,'dogs':len(dog_scores)})
    best=max(trials,key=lambda r:(r['correct'],-r['false_cat_dogs'],-r['subject_weight'],-int(r['tta'])))
    config={'architecture':'swin_t','checkpoint':'breed-swin-v2-fp16.pt',
            **{k:best[k] for k in ['tta','subject_weight','cat_threshold']},
            'checkpoint_sha256':hashlib.sha256(portable.read_bytes()).hexdigest(),
            'selected_on':'validation only:238 cats and500 dogs','species_negative_scope':'dogs',
            'selection_constraint':'at most2% false-cat among validation dogs; threshold>=0.5'}
    save_json(out/'selection.json',{'selected':best,'trials':trials,'config':config})
    save_json(ROOT/'configs/active-classifier.json',config)
    print('SELECTED',json.dumps(best),flush=True)


if __name__=='__main__':
    with ModelTask('select-swin-v2'):
        main()
