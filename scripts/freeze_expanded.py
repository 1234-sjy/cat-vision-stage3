"""Export storage weights or activate a validation-selected inference setting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json

BASE=ROOT/'artifacts/optimization-v3'
CHECKPOINT=ROOT/'weights/breed-swin-v3-20-fp16.pt'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def export(source=None):
    import torch
    source=source or ROOT/'weights/breed-swin-v3-generic.pt'
    if not (BASE/'training-generic/complete.json').exists(): raise RuntimeError('Training is not complete')
    if CHECKPOINT.exists(): raise FileExistsError('Retain existing exported weights')
    payload=torch.load(source,map_location='cpu',weights_only=True)
    payload={key:payload[key] for key in ['state_dict','classes','epoch','config','metrics']}
    payload['state_dict']={key:value.half() if value.is_floating_point() else value for key,value in payload['state_dict'].items()}
    payload['storage_dtype']='float16; runtime casts parameters to float32'
    with CHECKPOINT.open('xb') as f: torch.save(payload,f)
    save_json(BASE/'export.json',{'source':source.name,'source_sha256':sha(source),
        'export':CHECKPOINT.name,'export_sha256':sha(CHECKPOINT),'bytes':CHECKPOINT.stat().st_size,
        'epoch':payload['epoch'],'classes':payload['classes'],
        'image_size':payload['config'].get('image_size',224),
        'preprocess':payload['config'].get('preprocess','center'),
        'source_metrics':payload['metrics'],'source_config':payload['config']})
    print('EXPORTED',CHECKPOINT,flush=True)


def activate():
    original=json.loads((BASE/'validation-original/metrics.json').read_text(encoding='utf-8'))
    flip=json.loads((BASE/'validation-flip/metrics.json').read_text(encoding='utf-8'))
    if original['weights_sha256']!=sha(CHECKPOINT) or flip['weights_sha256']!=sha(CHECKPOINT):
        raise ValueError('Validation weight identities differ')
    tta=(flip['all']['accuracy'],flip['all']['macro_accuracy'])>(original['all']['accuracy'],original['all']['macro_accuracy'])
    storage=json.loads((BASE/'export.json').read_text(encoding='utf-8'))
    config={'architecture':'swin_t','checkpoint':CHECKPOINT.name,'classes':storage['classes'],
        'image_size':storage['image_size'],'preprocess':storage['preprocess'],
        'subject_weight':0.0,'tta':tta,'cat_threshold':0.5,'weight_sha256':sha(CHECKPOINT),
        'selection':'validation overall accuracy, then macro accuracy; no-flip wins ties; no test-guided settings'}
    old=ROOT/'configs/active-classifier.json'
    history=BASE/'active-classifier-before-v3.json'
    if history.exists(): raise FileExistsError('Activation history already exists')
    save_json(history,json.loads(old.read_text(encoding='utf-8')))
    save_json(old,config)
    save_json(BASE/'inference-selection.json',{'tta':tta,'original':original['all'],'flip':flip['all'],'test_used':False})
    from catvision.evaluation import inference_fingerprint
    save_json(ROOT/'configs/frozen-v3.json',{'active_classifier':config,
        'vlm':json.loads((ROOT/'configs/active-vlm.json').read_text(encoding='utf-8')),
        'shared_test_sha256':sha(BASE/'shared-test/manifest.json'),
        'expanded_test_sha256':sha(BASE/'data/test-manifest.json'),
        'fingerprints':{route:inference_fingerprint(ROOT,route) for route in ['traditional','vlm']},
        'frozen_before_test':True})
    print('ACTIVATED20', 'tta='+str(tta),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['export','activate'])
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.action=='export': export(args.source)
    else: activate()
