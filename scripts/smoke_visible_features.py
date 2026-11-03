"""Real sequential image observations; only frozen validation images are used."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE']='1'
from catvision.common import ROOT, load_image, save_json
from catvision.evaluation import read_verified_input
from catvision.run_control import ModelTask


if __name__ == '__main__':
    from catvision.vlm import LocalVLM
    with ModelTask('smoke-visible-features-v1'):
        out=ROOT/'artifacts/features-v1/validation-smoke'
        if out.exists():
            raise FileExistsError(out)
        rows=json.loads((ROOT/'artifacts/optimization-v4/data/val-selection.json').read_text(encoding='utf-8'))
        # The existing demo input is also a validation example.
        all_val=json.loads((ROOT/'artifacts/optimization-v4/data/val-full.json').read_text(encoding='utf-8'))
        fold=next(r for r in all_val if r['image_id']=='refined_Scottish Fold_286')
        selected=[fold,next(r for r in rows if r['breed']=='Bengal'),
                  next(r for r in rows if r['breed']=='Sphynx'),fold]
        results=[]
        engine=LocalVLM()
        try:
            for row in selected:
                content,sha=read_verified_input(row,ROOT.parent.parent/'work/cat-data')
                result=engine.predict(load_image(content,max_side=1600))
                results.append({'image_id':row['image_id'],'input_sha256':sha,'result':result})
                save_json(out/'results.json',results)
                print(json.dumps({'image_id':row['image_id'],'breed':result.get('breed'),
                    'features':result.get('features'),'feature_status':result.get('feature_status'),
                    'seconds':result['inference_seconds']},ensure_ascii=False),flush=True)
        finally:
            engine.close()
        if results[0]['result']['breed'] != results[-1]['result']['breed']:
            raise RuntimeError('Breed answer changed after descriptive passes')
        if any(len(row['result']['features'])!=2 for row in results):
            raise RuntimeError('A validation example did not return two observations')
        save_json(out/'complete.json',{'count':len(results),'same_image_breed_stable':True,
            'two_observations_each':True,'source':'validation only','teacher_breed_supplied_to_description':False})
