from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from .common import BREEDS, ROOT
from .geometry import crop_box, main_cat_index, mask_to_box
from .run_control import gpu_checkpoint, set_low_load_threads, synchronize_current_stream


def synchronize():
    synchronize_current_stream()


class TraditionalVision:
    def __init__(self, weights_dir: Path | None = None, classifier: bool = True):
        set_low_load_threads()
        from ultralytics import SAM, YOLO
        self.weights = weights_dir or ROOT / "weights"
        self.device = "0" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        self.detector = YOLO(str(self.weights / "yolo26n.pt"))
        self.segmenter = SAM(str(self.weights / "sam2.1_t.pt"))
        active = ROOT / 'configs/active-classifier.json'
        config = json.loads(active.read_text(encoding='utf-8')) if classifier and active.exists() else {}
        self.classifier_type = config.get('architecture','yolo26s-cls')
        if classifier and self.classifier_type == 'swin_t':
            from .breed_transformer import BreedTransformer
            self.classifier = BreedTransformer(self.weights/config['checkpoint'],config)
        else:
            self.classifier = YOLO(str(self.weights / "breed-best.pt")) if classifier else None
        self.load_seconds = time.perf_counter() - started

    def extract_subject(self, image: Image.Image) -> dict:
        synchronize()
        started = time.perf_counter()
        # PIL inputs are RGB; Ultralytics performs its own BGR conversion.
        detection = self.detector.predict(image, conf=0.15, imgsz=640, device=self.device,
                                          batch=1, workers=0, verbose=False)[0]
        synchronize()
        detection_seconds = time.perf_counter() - started
        pause_seconds = gpu_checkpoint()
        boxes = detection.boxes.xyxy.cpu().numpy()
        classes = detection.boxes.cls.cpu().numpy()
        cat_id = next(k for k, v in detection.names.items() if v == "cat")
        selected = main_cat_index(boxes, classes, cat_id)
        if selected is None:
            return {"status": "no_cat", "has_cat": False, "breed": None, "breed_zh": None,
                    "cat_count": 0, "detection_seconds": detection_seconds,
                    "inference_seconds": detection_seconds, "throttle_seconds": pause_seconds}
        box = boxes[selected].tolist()
        mask = None
        segment_error = None
        segment_started = time.perf_counter()
        try:
            segmented = self.segmenter.predict(image, bboxes=[box], device=self.device,
                                               imgsz=1024, batch=1, workers=0, verbose=False)[0]
            synchronize()
            segment_seconds = time.perf_counter() - segment_started
            pause_seconds += gpu_checkpoint()
            if segmented.masks is not None and len(segmented.masks.data):
                mask = segmented.masks.data[0].cpu().numpy() > 0.5
                if mask.shape != (image.height, image.width):
                    mask = cv2.resize(mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST).astype(bool)
                if mask_to_box(mask) is None:
                    mask = None
        except (RuntimeError, ValueError):
            # Stop this task. A GPU/runtime failure must never trigger fallback or retry.
            raise
        # Retain color/texture inside the subject crop; masks are not painted into classifier input.
        subject_box = mask_to_box(mask) if mask is not None else None
        # Reject masks whose occupied region is far smaller than the detection; preserve the box fallback.
        if subject_box and (subject_box[2] - subject_box[0]) * (subject_box[3] - subject_box[1]) < 0.1 * (box[2] - box[0]) * (box[3] - box[1]):
            subject_box = None
        region = crop_box(subject_box or box, image.size)
        return {"status": "ok", "has_cat": True, "box": box,
                "detection_score": float(detection.boxes.conf[selected]),
                "cat_count": int(sum(classes == cat_id)), "mask": mask,
                "crop": image.crop(region), "crop_box": list(region),
                "crop_source": "sam_mask" if subject_box else "detection_box",
                "segmentation_error": segment_error, "detection_seconds": detection_seconds,
                "segmentation_seconds": segment_seconds,
                "inference_seconds": max(0.0,time.perf_counter() - started - pause_seconds),
                "throttle_seconds": pause_seconds}

    def predict(self, image: Image.Image) -> dict:
        started = time.perf_counter()
        result = self.extract_subject(image)
        if self.classifier_type == 'swin_t':
            localization_status = result['status']
            extract_pause = result['throttle_seconds']
            prediction = self.classifier.predict(image,result.get('crop'))
            result.update(prediction)
            pause = extract_pause + self.classifier.last_throttle_seconds
            result.update(localization_status=localization_status,
                          classification_source='original_and_subject' if result.get('crop') is not None and self.classifier.config['subject_weight']>0 else 'original',
                          route='yolo26_sam2_swin_v2',throttle_seconds=pause,
                          inference_seconds=max(0.0,time.perf_counter()-started-pause))
            return result
        if result["status"] != "ok":
            return result
        if self.classifier is None:
            raise RuntimeError("Breed classifier has not been loaded")
        classification = self.classifier.predict(result["crop"], imgsz=320,
                                                  device=self.device, batch=1, workers=0, verbose=False)[0]
        classification_pause = gpu_checkpoint()
        breed = classification.names[classification.probs.top1]
        if breed not in BREEDS:
            raise ValueError(f"Unexpected classifier class: {breed}")
        synchronize()
        result.update(breed=breed, breed_zh=BREEDS[breed],
                      breed_score=float(classification.probs.top1conf),
                      inference_seconds=max(0.0, time.perf_counter() - started - result['throttle_seconds'] - classification_pause),
                      route="yolo26_sam2_classifier")
        result["throttle_seconds"] += classification_pause
        return result

    def close(self):
        self.detector = self.segmenter = self.classifier = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def visualization(image: Image.Image, result: dict) -> Image.Image:
    array = np.array(image).copy()
    mask = result.get("mask")
    if mask is not None:
        color = np.zeros_like(array)
        color[:] = (33, 155, 135)
        array[mask] = (0.75 * array[mask] + 0.25 * color[mask]).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(array, contours, -1, (33, 155, 135), 2)
    if "box" in result:
        x1, y1, x2, y2 = map(round, result["box"])
        cv2.rectangle(array, (x1, y1), (x2, y2), (215, 111, 62), 2)
    return Image.fromarray(array)
