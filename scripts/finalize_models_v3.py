"""Export validation-selected weights and configure one-breed ensemble output."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json

BASE=ROOT/'artifacts/optimization-v3'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    import torch
    selection=json.loads((BASE/'ensemble-validation/selected.json').read_text(encoding='utf-8'))
    choice=selection['selection'];configs=selection['configs']
    full=configs[choice['full']];detail=configs[choice['detail']]
    exports=[]
    for config,name in [(full,'breed-swin-v3-20-fp16.pt'),(detail,'breed-swin-v3-detail20-fp16.pt')]:
        source=Path(config['weights']);destination=ROOT/'weights'/name
        if destination.exists(): raise FileExistsError(destination)
        if sha(source)!=config['weights_sha256']: raise ValueError('Selected validation source changed')
        payload=torch.load(source,map_location='cpu',weights_only=True)
        compact={key:payload[key] for key in ['classes','epoch','config','metrics']}
        compact['state_dict']={key:value.half() if value.is_floating_point() else value
                               for key,value in payload['state_dict'].items()}
        compact['storage_dtype']='FP16 storage, FP32 runtime; optimizer/RNG retained in original checkpoint'
        with destination.open('xb') as f:
            torch.save(compact,f);f.flush();os.fsync(f.fileno())
        if destination.stat().st_size>=100*1024**2: raise ValueError('Export exceeds GitHub normal file limit')
        exports.append({'file':name,'sha256':sha(destination),'bytes':destination.stat().st_size,
            'source':str(source),'source_sha256':config['weights_sha256'],'source_epoch':payload['epoch'],
            'image_size':config['image_size'],'preprocess':config['preprocess']})
        del payload,compact
    old=ROOT/'configs/active-classifier.json'
    history=BASE/'active-classifier-before-v3.json'
    if history.exists(): raise FileExistsError(history)
    save_json(history,json.loads(old.read_text(encoding='utf-8')))
    active={'architecture':'swin_t','checkpoint':exports[0]['file'],'classes':full['classes'],
        'image_size':full['image_size'],'preprocess':full['preprocess'],'tta':full['tta'],
        'cat_threshold':.5,'subject_weight':0.,'weight_sha256':exports[0]['sha256'],
        'secondary_classifier':{'checkpoint':exports[1]['file'],'weight':choice['detail_weight'],
            'image_size':detail['image_size'],'preprocess':detail['preprocess'],'tta':detail['tta'],
            'weight_sha256':exports[1]['sha256']},
        'selection':'same frozen20-class validation set only; overall accuracy, then macro; no test used'}
    save_json(old,active)
    save_json(BASE/'final-model.json',{'exports':exports,'selection':choice,'active_classifier':active,
        'training_manifest_sha256':configs['full']['manifest_sha256'],
        'final_test_used_for_selection':False,'final_test_count':1872})
    print('EXPORTED AND CONFIGURED',[(r['file'],r['bytes']) for r in exports],flush=True)


if __name__=='__main__': main()
