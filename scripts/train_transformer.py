"""Low-load Swin finetuning. No test input, hidden downloads, retry or restart."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, gpu_checkpoint, require_training_enabled, set_low_load_threads


def save_checkpoint(path,payload):
    import torch
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        torch.save(payload,f)
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def main():
    import torch
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    from catvision.breed_transformer import BreedSwin, CLASSES, train_transform, eval_transform
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    require_training_enabled(); set_low_load_threads()
    seed=20261102
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=ROOT/'artifacts/optimization-v2/training'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'history.json').exists():
        raise FileExistsError('Existing training run retained; no automatic resume or overwrite')
    manifest=ROOT/'artifacts/optimization-v2/train-manifest.json'
    records=json.loads(manifest.read_text())
    if any(r['split'] not in {'train','val'} for r in records):
        raise ValueError('Only train/val sources allowed')
    class PetDataset(Dataset):
        def __init__(self,split):
            self.records=[r for r in records if r['split']==split]
            self.split=split
            self.transform=train_transform() if split=='train' else eval_transform()
        def __len__(self): return len(self.records)
        def __getitem__(self,index):
            r=self.records[index]
            subject=ROOT/'data/classification/train'/r['breed']/(r['image_id']+'-subject.jpg')
            path=subject if self.split=='train' and r['is_cat'] and random.random()<0.5 else args.data_root/r['relative_path']
            image=load_image(path)
            return self.transform(image),CLASSES.index(r['breed']) if r['is_cat'] else -1,int(r['is_cat'])
    train=DataLoader(PetDataset('train'),batch_size=8,shuffle=True,num_workers=0,pin_memory=False)
    val=DataLoader(PetDataset('val'),batch_size=8,shuffle=False,num_workers=0,pin_memory=False)
    pretrained=ROOT/'weights/swin_t-704ceda3.pth'
    source_sha=hashlib.sha256(pretrained.read_bytes()).hexdigest()
    if not source_sha.startswith('704ceda3'):
        raise ValueError('Unexpected generic pretrained weights')
    model=BreedSwin(pretrained).to(device)
    optimizer=torch.optim.AdamW([
        {'params':model.backbone.parameters(),'lr':1.5e-5,'base_lr':1.5e-5},
        {'params':list(model.breed_head.parameters())+list(model.species_head.parameters()),'lr':5e-4,'base_lr':5e-4}],
        weight_decay=1e-4)
    breed_loss=nn.CrossEntropyLoss(label_smoothing=0.05)
    species_loss=nn.CrossEntropyLoss()
    config={'architecture':'torchvision.swin_t','initialization_sha256':source_sha,
            'seed':seed,'epochs_max':args.epochs,'warmup_frozen_epochs':2,'patience':8,
            'batch':8,'validation_batch':8,'workers':0,'cpu_threads':2,'image_size':224,
            'backbone_lr':1.5e-5,'heads_lr':5e-4,'optimizer':'AdamW','weight_decay':1e-4,
            'breed_label_smoothing':0.05,'species_loss_weight':0.5,'precision':'FP32, TF32 matmul allowed',
            'pause_after_sync_seconds':0.05,'train_source_count':len(train.dataset),'val_source_count':len(val.dataset),
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'selection':'validation original-image breed correct, then fewer false-cat dogs',
            'no_test_input':True,'automatic_retry':False,'automatic_resume':False}
    save_json(out/'config.json',config)
    print('CONFIG',json.dumps(config),flush=True)
    history=[]; best=(-1,-10000); wait=0; started=time.perf_counter()
    for epoch in range(args.epochs):
        for p in model.backbone.parameters():
            p.requires_grad_(epoch>=2)
        factor=0.15+0.85*(1+math.cos(math.pi*max(0,epoch-2)/max(1,args.epochs-2)))/2
        for group in optimizer.param_groups: group['lr']=group['base_lr']*factor
        model.train(); total_loss=0; seen=0; e_start=time.perf_counter()
        for step,(images,breeds,species) in enumerate(train):
            images=images.to(device); breeds=breeds.to(device); species=species.to(device)
            optimizer.zero_grad(set_to_none=True)
            bp,sp=model(images)
            mask=breeds>=0
            loss_b=breed_loss(bp[mask],breeds[mask]) if mask.any() else bp.sum()*0
            loss=loss_b+0.5*species_loss(sp,species)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch{epoch+1} batch{step+1}; no retry')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); gpu_checkpoint(pause_seconds=0.05)
            total_loss+=float(loss.detach())*len(images); seen+=len(images)
            if (step+1)%60==0:
                print(f'TRAIN epoch={epoch+1} batch={step+1}/{len(train)} loss={total_loss/seen:.5f} allocatedMB={torch.cuda.memory_allocated()/1024**2:.0f}',flush=True)
        model.eval(); correct=0; cats=0; cats_accepted=0; dogs=0; false_cat=0
        with torch.inference_mode():
            for images,breeds,species in val:
                bp,sp=model(images.to(device)); bp=bp.cpu(); sp=sp.cpu()
                mask=species==1
                correct+=int((bp.argmax(1)[mask]==breeds[mask]).sum()); cats+=int(mask.sum())
                cats_accepted+=int(((sp.argmax(1)==1)&mask).sum())
                dogs+=int((~mask).sum()); false_cat+=int(((sp.argmax(1)==1)&~mask).sum())
                gpu_checkpoint(pause_seconds=0.05)
        row={'epoch':epoch+1,'train_loss':total_loss/seen,'val_breed_correct':correct,'val_cats':cats,
             'val_accuracy':correct/cats,'val_cats_accepted':cats_accepted,'val_false_cat_dogs':false_cat,
             'val_dogs':dogs,'seconds':time.perf_counter()-e_start,'elapsed_seconds':time.perf_counter()-started}
        history.append(row); save_json(out/'history.json',history)
        payload={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                 'classes':CLASSES,'epoch':epoch+1,'config':config,'metrics':row}
        epoch_path=out/f'epoch-{epoch+1:02d}.pt'
        save_checkpoint(epoch_path,payload)
        score=(correct,-false_cat)
        if score>best:
            best=score; wait=0
            save_checkpoint(ROOT/'weights/breed-swin-v2.pt',payload)
            save_json(out/'best.json',row)
        else: wait+=1
        print('EPOCH',json.dumps(row),'BEST',best,'patience',wait,flush=True)
        if wait>=8:
            print('EARLY STOP validation only',flush=True);break
    save_json(out/'complete.json',{'best':json.loads((out/'best.json').read_text()),
                                 'epochs_completed':len(history),'seconds':time.perf_counter()-started})
    print('TRAINING COMPLETED',flush=True)


if __name__=='__main__':
    with ModelTask('train-swin-v2'):
        main()
