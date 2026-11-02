"""Validation-only original/subject ablation; no test-set tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, gpu_checkpoint, set_low_load_threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    set_low_load_threads()
    import torch
    from ultralytics import YOLO
    records = [r for r in json.loads((ROOT/'artifacts/data/manifest.json').read_text()) if r['split']=='val']
    views = {r['image_id']: r for r in map(json.loads, (ROOT/'artifacts/views/records.jsonl').read_text().splitlines())}
    model = YOLO(str(ROOT/'weights/breed-best.pt'))
    output = ROOT/'artifacts/optimization-v2/yolo-validation-ablation.json'
    rows = []
    for i, r in enumerate(records):
        image = load_image(args.data_root/r['relative_path'])
        view = views[r['image_id']]
        crop = image.crop(view['crop_box']) if view['status']=='ok' else image
        result = {'image_id':r['image_id'], 'true_breed':r['breed'], 'detected':view['status']=='ok'}
        for name, im in [('original',image),('subject',crop)]:
            p = model.predict(im, imgsz=320, batch=1, workers=0, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
            gpu_checkpoint(pause_seconds=0.2)
            result[name] = {p.names[j]:float(v) for j,v in enumerate(p.probs.data.cpu().tolist())}
        rows.append(result)
        if (i+1)%20==0 or i+1==len(records):
            save_json(output,rows)
            print(f'ABLATION {i+1}/{len(records)}',flush=True)
    for w in [0,0.25,0.5,0.75,1]:
        correct=sum(max(r['original'],key=lambda b:(1-w)*r['original'][b]+w*r['subject'][b])==r['true_breed'] for r in rows)
        print(f'SUBJECT_WEIGHT {w}: {correct}/{len(rows)} = {correct/len(rows):.4%}',flush=True)
    print('HARD_GATE',sum(r['detected'] and max(r['subject'],key=r['subject'].get)==r['true_breed'] for r in rows),len(rows),flush=True)


if __name__=='__main__':
    with ModelTask('validation-ablation'):
        main()
