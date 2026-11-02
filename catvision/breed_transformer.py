"""Independently trained Swin features, with separate breed and species heads."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from .common import BREEDS, OXFORD_BREEDS, ROOT
from .run_control import gpu_checkpoint, set_low_load_threads

CLASSES = list(OXFORD_BREEDS)  # Original-v2 scripts retain their historical 12-class order.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def checkpoint_classes(payload: dict, expected: list[str] | None = None) -> list[str]:
    names = payload.get('classes')
    if (not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names)
            or len(set(names)) != len(names) or any(n not in BREEDS for n in names)):
        raise ValueError('Invalid checkpoint class mapping')
    if expected is not None and names != expected:
        raise ValueError('Checkpoint classes disagree with active configuration')
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


class Letterbox:
    def __init__(self,size): self.size=size
    def __call__(self,image):
        from PIL import Image, ImageOps
        return ImageOps.pad(image,(self.size,self.size),method=Image.Resampling.BICUBIC,
                            color=tuple(round(v*255) for v in MEAN))


def eval_transform(image_size=224,preprocess='center'):
    if preprocess=='letterbox':
        spatial=[Letterbox(image_size)]
    elif preprocess=='center':
        spatial=[transforms.Resize(round(image_size*256/224),interpolation=InterpolationMode.BICUBIC),
                 transforms.CenterCrop(image_size)]
    else:
        raise ValueError('Unknown image preprocessing')
    return transforms.Compose(spatial+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


def train_transform(image_size=224,preprocess='center'):
    if preprocess=='letterbox':
        return transforms.Compose([Letterbox(image_size),transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.12,0.12,0.12,0.03),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size,scale=(0.7,1.0),ratio=(0.8,1.25),interpolation=InterpolationMode.BICUBIC),
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
        self.config={'subject_weight':0.5,'tta':True,'cat_threshold':0.5,**(config or {})}
        weights=Path(weights) if weights is not None else ROOT/'weights/breed-swin-v2.pt'
        secondary=self.config.get('secondary_classifier')
        if secondary is not None:
            if not isinstance(secondary,dict):
                raise ValueError('Secondary classifier must be a configuration object')
            name=secondary.get('checkpoint')
            if (not isinstance(name,str) or name in {'','.','..'} or Path(name).name != name
                    or '/' in name or '\\' in name or ':' in name):
                raise ValueError('Secondary checkpoint must be a filename inside weights')
            alpha=float(secondary['weight'])
            if not 0 <= alpha <= 1:
                raise ValueError('Secondary weight must be between zero and one')
        payload=torch.load(weights,map_location='cpu',weights_only=True)
        self.classes=checkpoint_classes(payload,expected=self.config.get('classes'))
        self.model=BreedSwin(classes=self.classes)
        self.model.load_state_dict(payload['state_dict'])
        self.model.float().to(self.device).eval()
        self.transform=eval_transform(self.config.get('image_size',224),self.config.get('preprocess','center'))
        self.components=[(self.model,self.transform,1.0)]
        self.component_tta=[self.config['tta']]
        if secondary is not None:
            payload=torch.load(weights.parent/name,map_location='cpu',weights_only=True)
            checkpoint_classes(payload,expected=self.classes)
            model=BreedSwin(classes=self.classes)
            model.load_state_dict(payload['state_dict'])
            model.float().to(self.device).eval()
            transform=eval_transform(secondary.get('image_size',224),secondary.get('preprocess','center'))
            self.components=[(self.model,self.transform,1-alpha),(model,transform,alpha)]
            self.component_tta.append(secondary.get('tta',False))
        self.last_throttle_seconds=0

    @torch.inference_mode()
    def probabilities_batch(self,images: list,tta: bool | None = None) -> dict:
        """Fuse FP32 softmax probabilities, retaining the input image order."""
        from PIL import ImageOps
        if not images:
            return {'breed_probs':[],'cat_probability':[]}
        breeds=torch.zeros((len(images),len(self.classes)),dtype=torch.float32)
        presence=torch.zeros(len(images),dtype=torch.float32)
        for (model,transform,weight),configured_tta in zip(self.components,self.component_tta,strict=True):
            use_flip=configured_tta if tta is None else tta
            flips=(False,True) if use_flip else (False,)
            for flip in flips:
                # Each component starts from the original PIL source, with its own spatial transform.
                tensor=torch.stack([transform(ImageOps.mirror(image) if flip else image)
                                    for image in images]).to(self.device,dtype=torch.float32)
                breed,cat=model(tensor)
                breeds.add_(breed.float().softmax(-1).cpu(),alpha=weight/len(flips))
                presence.add_(cat.float().softmax(-1)[:,1].cpu(),alpha=weight/len(flips))
                self.last_throttle_seconds+=gpu_checkpoint(pause_seconds=0.2)
        return {'breed_probs':breeds.tolist(),'cat_probability':presence.tolist()}

    def probabilities(self,image,tta: bool | None = None) -> dict:
        batch=self.probabilities_batch([image],tta=tta)
        return {'breed_probs':batch['breed_probs'][0],'cat_probability':batch['cat_probability'][0]}

    def predict(self,image,subject=None):
        self.last_throttle_seconds=0
        original=self.probabilities(image)
        subject_p=self.probabilities(subject) if subject is not None and self.config['subject_weight']>0 else None
        return decision(combine_views(original,subject_p,self.config['subject_weight']),
                        self.classes,self.config['cat_threshold'])
