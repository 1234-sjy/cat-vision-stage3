from __future__ import annotations

import gc
from contextlib import nullcontext
import json
from pathlib import Path
import time

from PIL import Image
import torch

from .common import BREEDS, ROOT, parse_vlm_output
from .run_control import gpu_checkpoint, set_low_load_threads, synchronize_current_stream

PROMPT_VERSION = "cat-breed-v1"
PROMPT = """观察图片中的猫。若有多只猫，只判断画面中面积最大的猫。
任务是根据可见外观识别一个猫品种。候选代码与中文名如下：
""" + "\n".join(f"{code}: {name}" for code, name in BREEDS.items()) + """
只返回一个JSON对象，不要Markdown。字段如下：
has_cat: 布尔值，图片里是否有真实的猫；
breed: 上述候选代码中的一个。没有猫时用null，明确属于其他品种时用"other"；
features: 最多3条简短中文可见外观特征，如耳朵、脸型、毛长、花纹或眼睛。
不要给多个候选，不要编造百分比，不要根据图片文件名判断。
示例结构：{"has_cat":true,"breed":"候选代码","features":["可见特征"]}。
"""

SETTINGS_PATH = ROOT / 'configs/active-vlm.json'
SETTINGS = json.loads(SETTINGS_PATH.read_text(encoding='utf-8')) if SETTINGS_PATH.exists() else {}
if SETTINGS:
    from .vlm_prompts import breed_prompt
    PROMPT_VERSION = SETTINGS['prompt_version']
    PROMPT = breed_prompt(PROMPT_VERSION)


class LocalVLM:
    def __init__(self, model_dir: Path | None = None, *, accelerate_patch=True,
                 prompt=None, prompt_version=None, max_pixels=None, max_new_tokens=None,
                 output_format=None, adapter_dir=None):
        set_low_load_threads()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("本地4B视觉大模型需要CUDA显卡，请先安装对应PyTorch")
        location = model_dir or ROOT / "weights/qwen3-vl-4b"
        max_pixels = SETTINGS.get('max_pixels',768*768) if max_pixels is None else max_pixels
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            location, local_files_only=True, min_pixels=128 * 128, max_pixels=max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            location, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa")
        self.model.eval()
        self.accelerate_patch = accelerate_patch
        if accelerate_patch:
            from .vlm_acceleration import LinearPatchProjection
            self.model.model.visual.patch_embed = LinearPatchProjection(self.model.model.visual.patch_embed.proj)
        from .vlm_adapter import resolve_adapter
        adapter = (resolve_adapter(ROOT, SETTINGS) if adapter_dir is None
                   else None if adapter_dir is False else Path(adapter_dir))
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
            self.model.eval()
        self.adapter = adapter.relative_to(ROOT).as_posix() if adapter is not None else None
        self.prompt = PROMPT if prompt is None else prompt
        self.prompt_version = PROMPT_VERSION if prompt_version is None else prompt_version
        self.max_new_tokens = SETTINGS.get('max_new_tokens',192) if max_new_tokens is None else max_new_tokens
        self.output_format = SETTINGS.get('output_format','json') if output_format is None else output_format
        feature_path = ROOT/'configs/feature-description.json'
        self.feature_settings = json.loads(feature_path.read_text(encoding='utf-8')) if feature_path.exists() else {}
        if self.output_format not in {'json','lines'}:
            raise ValueError('Unsupported visual-model output format')
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.top_k = 50
        self.load_seconds = time.perf_counter() - started

    def predict(self, image: Image.Image) -> dict:
        synchronize_current_stream()
        started = time.perf_counter()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": self.prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        answer = generated[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(answer, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        synchronize_current_stream()
        inference_seconds = time.perf_counter() - started
        pause_seconds = gpu_checkpoint()
        if self.output_format == 'lines':
            from .vlm_format import parse_breed_lines
            result = parse_breed_lines(text)
        else:
            result = parse_vlm_output(text)
        classification_seconds = inference_seconds
        if self.feature_settings.get('enabled'):
            result.update(self.describe(image))
            inference_seconds = time.perf_counter()-started
        result.update(route="qwen3_vl_4b", model="Qwen/Qwen3-VL-4B-Instruct",
                      adapter=self.adapter,
                      classification_seconds=classification_seconds,
                      prompt_version=self.prompt_version, inference_seconds=inference_seconds,
                      patch_projection='equivalent_linear' if self.accelerate_patch else 'original_conv3d',
                      output_format=self.output_format,
                      throttle_seconds=pause_seconds,
                      input_tokens=int(inputs["input_ids"].shape[1]), output_tokens=int(answer.shape[1]))
        return result

    def _image_text(self, image, prompt, max_new_tokens):
        """Called in the request queue; never alongside a second adapter operation."""
        synchronize_current_stream()
        started=time.perf_counter()
        messages=[{'role':'user','content':[
            {'type':'image','image':image},{'type':'text','text':prompt}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.model.device)
        # Context manager restores the class adapter even if generation raises.
        context=self.model.disable_adapter() if self.adapter else nullcontext()
        with context,torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            generated=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        answer=generated[:,inputs['input_ids'].shape[1]:]
        text=self.processor.batch_decode(answer,skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        synchronize_current_stream()
        return text,{'seconds':time.perf_counter()-started,
                     'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':int(answer.shape[1])}

    def describe(self, image):
        from .vlm_features import OBSERVATION_PROMPT,OBSERVATION_PROMPT_VERSION,parse_observation
        text,usage=self._image_text(image,OBSERVATION_PROMPT,self.feature_settings.get('max_new_tokens',384))
        result=parse_observation(text)
        result.update(observation_prompt_version=OBSERVATION_PROMPT_VERSION,
            observation_seconds=usage['seconds'],observation_output_tokens=usage['output_tokens'],
            observation_adapter_disabled=bool(self.adapter),
            observation_input='same_original_image_without_breed_answer')
        return result

    def answer_question(self, image, question):
        from .vlm_features import QUESTION_PROMPT_VERSION,image_question_prompt,split_image_questions
        question=question.strip()
        if not 1<=len(question)<=500:
            raise ValueError('图片问题应为1至500个字符')
        parts=split_image_questions(question)
        replies=[]
        for part in parts:
            text,usage=self._image_text(image,image_question_prompt(part),
                self.feature_settings.get('question_max_new_tokens',256))
            replies.append({'question':part,'answer':text.strip(),'raw_text':text,**usage})
        answer=('\n'.join(f"{i+1}. {r['answer']}" for i,r in enumerate(replies))
                if len(replies)>1 else replies[0]['answer'])
        return {'status':'ok' if all(r['answer'] for r in replies) else 'partial',
            'question':question,'answer':answer,'subquestions':replies,
            'raw_text':'\n'.join(r['raw_text'] for r in replies),
            'inference_seconds':sum(r['seconds'] for r in replies),
            'input_tokens':sum(r['input_tokens'] for r in replies),
            'output_tokens':sum(r['output_tokens'] for r in replies),
            'prompt_version':QUESTION_PROMPT_VERSION,'adapter_disabled':bool(self.adapter),
            'model':'Qwen/Qwen3-VL-4B-Instruct','input':'same_original_image_and_question'}

    def close(self):
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
