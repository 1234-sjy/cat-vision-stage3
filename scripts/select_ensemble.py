"""Validation-only selection of complementary full-frame/detail classifiers."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from scripts.assess_expanded import score_group, classify_probabilities


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',type=Path,required=True)
    parser.add_argument('--full-flip',type=Path,required=True)
    parser.add_argument('--detail',type=Path,required=True)
    parser.add_argument('--detail-flip',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    folders={'full':args.full,'full_flip':args.full_flip,'detail':args.detail,'detail_flip':args.detail_flip}
    rows={name:read(path/'predictions.json') for name,path in folders.items()}
    configs={name:read(path/'config.json') for name,path in folders.items()}
    names=configs['full']['classes'];zh=configs['full']['classes_zh']
    for name in rows:
        if configs[name]['set']!='val' or configs[name]['classes']!=names: raise ValueError('Validation/class mapping mismatch')
        if {r['image_id']:r['input_sha256'] for r in rows[name]}!={r['image_id']:r['input_sha256'] for r in rows['full']}:
            raise ValueError('Must compare identical verified validation images')
    candidates=[];best=None;best_rows=None;best_score=None
    args.output.mkdir(parents=True)
    for full_name in ['full','full_flip']:
        for detail_name in ['detail','detail_flip']:
            secondary={r['image_id']:r for r in rows[detail_name]}
            for weight in [0,.15,.25,.4,.5]:
                merged=[]
                for row in rows[full_name]:
                    other=secondary[row['image_id']]
                    probabilities=[(1-weight)*a+weight*b for a,b in zip(row['breed_probabilities'],other['breed_probabilities'],strict=True)]
                    cat=(1-weight)*row['cat_probability']+weight*other['cat_probability']
                    prediction={**row,**classify_probabilities(probabilities,cat,names,zh)}
                    merged.append(prediction)
                metric=score_group(merged,names)
                candidate={'full':full_name,'detail':detail_name,'detail_weight':weight,
                    'correct':metric['correct'],'count':metric['count'],'accuracy':metric['accuracy'],
                    'macro_accuracy':metric['macro_accuracy'],'dog_false_cat':metric['dog_false_cat']}
                candidates.append(candidate)
                score=(metric['accuracy'],metric['macro_accuracy'],-metric['dog_false_cat'],
                    weight==0,-int(full_name.endswith('flip'))-int(detail_name.endswith('flip')),-weight)
                if best_score is None or score>best_score:
                    best_score=score;best=candidate;best_rows=merged
    save_json(args.output/'candidates.json',candidates)
    save_json(args.output/'selected.json',{'selection':best,'configs':configs,
        'test_used':False,'criterion':'overall validation accuracy, macro accuracy, fewer false cats; prefer simpler ties'})
    save_json(args.output/'predictions.json',best_rows)
    save_json(args.output/'metrics.json',score_group(best_rows,names))
    print(json.dumps(best,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
