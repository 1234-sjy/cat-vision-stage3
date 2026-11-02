"""Validation-only instrumentation; each result is freshly inferred, no retries."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['CATVISION_FAST_INFERENCE'] = '1'
from catvision.common import ROOT, load_image, save_json
from catvision.run_control import ModelTask, synchronize_current_stream


def main():
    import torch
    from catvision.vlm import LocalVLM
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    records = json.loads((ROOT/'artifacts/optimization-v2/train-manifest.json').read_text())
    chosen = []
    for breed in ['British_Shorthair', 'Bengal', 'Maine_Coon']:
        chosen.append(next(r for r in records if r['split']=='val' and r['breed']==breed))
    engine = LocalVLM()
    calls = []
    forward = engine.model.forward
    def timed_forward(*a, **kw):
        synchronize_current_stream(); started = time.perf_counter()
        result = forward(*a, **kw)
        synchronize_current_stream()
        ids = kw.get('input_ids')
        calls.append({'tokens': int(ids.shape[1]) if ids is not None else None,
                      'seconds': time.perf_counter()-started})
        return result
    engine.model.forward = timed_forward
    vision = []
    def before(module, inputs):
        synchronize_current_stream(); module._profile_started = time.perf_counter()
    def after(module, inputs, output):
        synchronize_current_stream(); vision.append(time.perf_counter()-module._profile_started)
    visual = engine.model.model.visual
    h1=visual.register_forward_pre_hook(before); h2=visual.register_forward_hook(after)
    parts = {}
    hooks = []
    for name, module in [('patch_projection', visual.patch_embed), *[(f'block_{i}', block) for i,block in enumerate(visual.blocks)]]:
        hooks.append(module.register_forward_pre_hook(before))
        def part_after(module, inputs, output, name=name):
            synchronize_current_stream()
            parts[name] = parts.get(name,0) + time.perf_counter()-module._profile_started
        hooks.append(module.register_forward_hook(part_after))
    output = {'torch':torch.__version__, 'cuda':torch.version.cuda,
              'load_seconds':engine.load_seconds,'threads':torch.get_num_threads(),
              'device_map':str(getattr(engine.model,'hf_device_map',None)),
              'dtype':str(engine.model.dtype),'generation_config':engine.model.generation_config.to_dict(), 'runs':[]}
    for row in chosen:
        calls.clear(); vision.clear(); parts.clear()
        result=engine.predict(load_image(args.data_root/row['relative_path']))
        output['runs'].append({'image_id':row['image_id'],'true_breed':row['breed'],
                               'result':result,'forward_calls':calls.copy(),'vision_seconds':vision.copy(),
                               'vision_components':parts.copy()})
        save_json(args.output, output)
        print(json.dumps(output['runs'][-1],ensure_ascii=False),flush=True)
    h1.remove();h2.remove()
    for hook in hooks: hook.remove()
    engine.close()


if __name__=='__main__':
    with ModelTask('vlm-profile-v3'): main()
