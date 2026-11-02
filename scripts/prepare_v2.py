"""Independent dog negatives and a new held-out comparison, fixed before training."""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.data import audit_records, choose_comparison


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    out=ROOT/'artifacts/optimization-v2'
    if (out/'train-manifest.json').exists() or (out/'comparison-v2.json').exists():
        raise FileExistsError('Frozen manifests already exist; refusing to replace them')
    cats=json.loads((ROOT/'artifacts/data/manifest.json').read_text())
    old=json.loads((ROOT/'artifacts/data/comparison.json').read_text())
    old_ids={r['image_id'] for r in old}
    new=choose_comparison([r for r in cats if r['image_id'] not in old_ids],seed=20261102)
    groups=defaultdict(list)
    for line in (args.data_root/'annotations/trainval.txt').read_text().splitlines():
        fields=line.split()
        if fields and not fields[0].startswith('#') and int(fields[2])==2:
            groups[fields[0].rsplit('_',1)[0]].append(fields[0])
    previous= random.Random(17)
    rng=random.Random(20261102)
    dogs=[]
    forbidden={r['sha256'] for r in cats}
    for breed in sorted(groups):
        options=sorted(groups[breed])
        existing_val=previous.sample(options,2)
        remaining=[x for x in options if x not in existing_val]
        rng.shuffle(remaining)
        val=existing_val+remaining[:18]
        train=remaining[18:58]
        for split,ids in [('val',val),('train',train)]:
            for image_id in ids:
                path=args.data_root/f'images/{image_id}.jpg'
                image=load_image(path)
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                if sha in forbidden:
                    raise ValueError(f'Duplicate selected negative: {image_id}')
                forbidden.add(sha)
                dogs.append({'image_id':image_id,'relative_path':f'images/{image_id}.jpg',
                             'breed':'negative_dog','dog_breed':breed,'split':split,
                             'width':image.width,'height':image.height,'sha256':sha})
    sources=[dict(r,is_cat=True) for r in cats if r['split'] in {'train','val'}]
    sources += [dict(r,is_cat=False) for r in dogs]
    audit_records(sources)
    save_json(out/'train-manifest.json',sources)
    save_json(out/'comparison-v2.json',new)
    save_json(out/'data-summary.json',{
        'source':'Oxford-IIIT Pet official trainval/test',
        'cats':dict(Counter(r['split'] for r in sources if r['is_cat'])),
        'dogs':dict(Counter(r['split'] for r in sources if not r['is_cat'])),
        'comparison_seed':20261102,'comparison_images':len(new),
        'overlap_with_old_comparison':len(old_ids & {r['image_id'] for r in new}),
        'negative_scope':'dogs only; no claim of universal open-world rejection',
        'dog_policy':'20 validation per breed including previous seed17 negatives;40 training per breed'})
    print(json.dumps(json.loads((out/'data-summary.json').read_text()),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
