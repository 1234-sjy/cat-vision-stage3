"""Record model, policy and independent test identity before test inference."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT,save_json
from catvision.evaluation import inference_fingerprint

if __name__=='__main__':
    target=ROOT/'configs/frozen-v2.json'
    if target.exists():raise FileExistsError('Frozen experiment already exists')
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    active=read(ROOT/'configs/active-classifier.json')
    checkpoint=ROOT/'weights'/active['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=active['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    manifest=ROOT/'artifacts/optimization-v2/comparison-v2.json'
    records=read(manifest)
    ids={r['image_id'] for r in records}
    trainval={r['image_id'] for r in read(ROOT/'artifacts/optimization-v2/train-manifest.json')}
    old={r['image_id'] for r in read(ROOT/'artifacts/data/comparison.json')}
    if len(ids)!=120 or ids&trainval or ids&old:
        raise ValueError('New test must be120 distinct held-out images outside old120')
    save_json(target,{'created_utc':datetime.now(timezone.utc).isoformat(),'active_classifier':active,
                     'traditional_fingerprint':inference_fingerprint(ROOT,'traditional'),
                     'vlm_fingerprint':inference_fingerprint(ROOT,'vlm'),
                     'comparison_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     'comparison_images':120,'selection_complete_before_testing':True,
                     'vlm_configuration':{k:v for k,v in read(ROOT/'configs/frozen-experiment.json').items()
                                          if k.startswith('vlm') or k in {'max_new_tokens','vlm_do_sample'}},
                     'test_policy':'No changes in model, fusion, threshold, prompt or generation after this freeze'})
    print('V2 FROZEN BEFORE TEST PREDICTIONS',flush=True)
