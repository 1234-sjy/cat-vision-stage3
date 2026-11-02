"""Fresh serial/parallel timings on validation inputs; no historical result replay."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE']='1'
from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, GpuReservation
from catvision.server import LocalComparison


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    rows=json.loads((ROOT/'artifacts/optimization-v2/train-manifest.json').read_text(encoding='utf-8'))
    chosen=[next(r for r in rows if r['split']=='val' and r['breed']==b) for b in ['British_Shorthair','Bengal','Maine_Coon']]
    pipeline=LocalComparison(keep_loaded=True)
    pipeline.preload()
    records=[]
    with ModelTask('benchmark-parallel-v3'):
        for index,row in enumerate(chosen):
            image=load_image(args.data_root/row['relative_path'])
            # Alternate order so a warmer GPU does not always favour one mode.
            for mode in (['serial','parallel'] if index%2==0 else ['parallel','serial']):
                folder=args.output.parent/f'{args.output.stem}-runs'/(str(index)+'-'+mode)
                folder.mkdir(parents=True,exist_ok=False)
                torch.cuda.reset_peak_memory_stats()
                if mode=='parallel':
                    result=pipeline._run(image,folder,lambda text:None)
                else:
                    result={};started=time.perf_counter()
                    for route in ['traditional','vlm']:
                        engine,_=pipeline._engine_for(route)
                        result[route]=engine.predict(image.copy())
                        result[route].pop('crop',None);result[route].pop('mask',None)
                    result['total_seconds']=time.perf_counter()-started
                result.update(mode=mode,image_id=row['image_id'],scope='Oxford validation; v2 classifier with expanded VLM candidates',
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
                records.append(result);save_json(args.output,records)
                print(mode,row['image_id'],round(result['total_seconds'],3),flush=True)
        pipeline.close()


if __name__=='__main__':
    with GpuReservation(): main()
