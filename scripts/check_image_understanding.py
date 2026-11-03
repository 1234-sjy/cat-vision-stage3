"""Record real observations/questions or a fixed classification regression. No retries."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE']='1'
from catvision.common import ROOT,load_image,save_json
from catvision.evaluation import read_verified_input,score_predictions
from catvision.run_control import ModelTask


def main(args):
    from catvision.vlm import LocalVLM
    if args.output.exists():
        raise FileExistsError('Existing output is preserved; use a new directory after reviewing it')
    data=ROOT/'artifacts/optimization-v4/data'
    if args.mode=='validation':
        rows=json.loads((data/'val-full.json').read_text(encoding='utf-8'))
        first=next(r for r in rows if r['image_id']=='refined_Scottish Fold_286')
        selected=[first]
        for breed in ['Sphynx','Bengal','Maine_Coon','Persian']:
            selected.append(next(r for r in rows if r['breed']==breed))
        selected.append(next(r for r in rows if r.get('is_cat') is False))
        selected.append(first)
    else:
        selected=json.loads((data/'shared-regression.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    save_json(args.output/'selection.json',{'mode':args.mode,'selection':'fixed before reading outputs',
        'records':selected,'classification_config':json.loads((ROOT/'configs/active-vlm.json').read_text(encoding='utf-8')),
        'observation_config':json.loads((ROOT/'configs/feature-description.json').read_text(encoding='utf-8'))})
    predictions=[];engine=LocalVLM();started=time.perf_counter()
    try:
        for index,row in enumerate(selected):
            content,sha=read_verified_input(row,args.data_root)
            image=load_image(content,max_side=1600)
            result=engine.predict(image)
            result.update(image_id=row['image_id'],input_sha256=sha,true_breed=row['breed'])
            if args.mode=='validation':
                questions=['耳朵有什么特点？它是什么姿态？',
                    '它身上主要能看到毛发还是皮肤？',
                    '它现在是什么姿态？背景里有哪些物品？',
                    '这张图中有几只动物？它在哪里？',
                    '它的毛发是什么颜色和长短？',
                    '图中是猫还是其他动物？它在做什么？',
                    '耳朵有什么特点？它是什么姿态？']
                question=questions[index]
                result['question_result']=engine.answer_question(image,question)
                image.save(args.output/f'input-{index+1:02d}.png')
            predictions.append(result)
            save_json(args.output/'predictions.json',predictions)
            print(json.dumps({'index':index+1,'total':len(selected),'image_id':row['image_id'],
                'breed':result.get('breed'),'observation_status':result.get('observation_status'),
                'seconds':result.get('inference_seconds')},ensure_ascii=False),flush=True)
    finally:
        engine.close()
    complete={'count':len(predictions),'seconds':time.perf_counter()-started,
        'structured_observations':sum(p['observation_status']=='ok' for p in predictions),
        'classification_accuracy_is_not_description_accuracy':True}
    if args.mode=='regression':
        old=json.loads((ROOT/'artifacts/optimization-v4/shared-regression/predictions.json').read_text(encoding='utf-8'))
        lookup={r['image_id']:r for r in old}
        changed=[{'image_id':p['image_id'],'old':lookup[p['image_id']].get('breed'),'new':p.get('breed')}
                 for p in predictions if p.get('breed')!=lookup[p['image_id']].get('breed')]
        complete.update(metrics=score_predictions(selected,predictions),changed_classifications=changed,
            comparison='previous batch4 classification vs current individual prediction plus observation')
    else:
        complete['repeated_first_image_breed_stable']=predictions[0].get('breed')==predictions[-1].get('breed')
    save_json(args.output/'complete.json',complete)
    print(json.dumps(complete,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['validation','regression'],required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with ModelTask('image-understanding-'+args.mode):
        main(args)
