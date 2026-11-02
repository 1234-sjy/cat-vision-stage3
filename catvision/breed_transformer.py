"""Independently trained Swin features, with separate breed and species heads."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from .common import BREEDS, ROOT
from .run_control import gpu_checkpoint, set_low_load_threads

CLASSES = list(BREEDS)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def checkpoint_classes(payload: dict) -> list[str]:
    names = payload.get('classes')
    if (not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names)
            or len(set(names)) != len(names) or any(n not in BREEDS for n in names)):
        raise ValueError('Invalid checkpoint class mapping')
    return names.copy()


def combine_views(original: dict, subject: dict | None, subject_weight: float) -> dict:
    if not 0 <= subject_weight <= 1:
        raise ValueError('Subject weight must be between zero and one')
    if subject is None:
        return original.copy()
    return {'breed_probs':[(1-subject_weight)*a+subject_weight*b
                           for a,b in zip(original['breed_probs'],subject['breed_probs'],strict=True)],
            # Presence is judged on the full scene. A cropped dog's fur must not bypass rejection.
            'cat_probability':original['cat_probability']}


def decision(probabilities: dict, names: list[str], cat_threshold: float) -> dict:
    presence=float(probabilities['cat_probability'])
    if presence < cat_threshold:
        return {'status':'no_cat','has_cat':False,'breed':None,'breed_zh':None,
                'cat_probability':presence}
    index=max(range(len(names)),key=lambda i:probabilities['breed_probs'][i])
    breed=names[index]
    return {'status':'ok','has_cat':True,'breed':breed,'breed_zh':BREEDS.get(breed,breed),
            'breed_score':float(probabilities['breed_probs'][index]),'cat_probability':presence}


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.7,1.0),ratio=(0.8,1.25),interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.12,0.12,0.12,0.03),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


class BreedSwin(nn.Module):
    def __init__(self, pretrained_path: Path | None = None, classes: list[str] | None = None):
        super().__init__()
        self.backbone=models.swin_t(weights=None)
        if pretrained_path is not None:
            self.backbone.load_state_dict(torch.load(pretrained_path,map_location='cpu',weights_only=True))
        features=self.backbone.head.in_features
        self.backbone.head=nn.Identity()
        self.classes=checkpoint_classes({'classes':CLASSES if classes is None else classes})
        self.breed_head=nn.Linear(features,len(self.classes))
        self.species_head=nn.Linear(features,2)  # 0 dog, 1 cat

    def forward(self, x):
        features=self.backbone(x)
        return self.breed_head(features),self.species_head(features)


class BreedTransformer:
    def __init__(self, weights: Path | None = None, config: dict | None = None):
        set_low_load_threads()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        payload=torch.load(weights or ROOT/'weights/breed-swin-v2.pt',map_location='cpu',weights_only=True)
        self.classes=checkpoint_classes(payload)
        self.model=BreedSwin(classes=self.classes)
        self.model.load_state_dict(payload['state_dict'])
        self.model.to(self.device).eval()
        self.transform=eval_transform()
        self.config=config or {'subject_weight':0.5,'tta':True,'cat_threshold':0.5}
        self.last_throttle_seconds=0

    @torch.inference_mode()
    def probabilities(self,image,tta: bool | None = None) -> dict:
        from PIL import ImageOps
        use_flip=self.config['tta'] if tta is None else tta
        images=[image,ImageOps.mirror(image)] if use_flip else [image]
        breeds=[]; species=[]
        for view in images:
            tensor=self.transform(view).unsqueeze(0).to(self.device)
            breed,cat=self.model(tensor)
            breeds.append(breed.softmax(-1).float().cpu())
            species.append(cat.softmax(-1).float().cpu())
            self.last_throttle_seconds+=gpu_checkpoint(pause_seconds=0.2)
        return {'breed_probs':torch.cat(breeds).mean(0).tolist(),
                'cat_probability':float(torch.cat(species).mean(0)[1])}

    def predict(self,image,subject=None):
        self.last_throttle_seconds=0
        original=self.probabilities(image)
        subject_p=self.probabilities(subject) if subject is not None and self.config['subject_weight']>0 else None
        return decision(combine_views(original,subject_p,self.config['subject_weight']),
                        self.classes,self.config['cat_threshold'])
