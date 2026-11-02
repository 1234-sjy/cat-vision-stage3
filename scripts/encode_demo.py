"""Encode actual browser screenshots; preserve capture timing and model displays."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT,save_json


if __name__=='__main__':
    import imageio_ffmpeg
    directory=ROOT/'artifacts/demo-v2'
    recording=json.loads((directory/'recording.json').read_text())
    frames=recording['frames']
    target=directory/'实际运行演示.mp4'
    if target.exists():raise FileExistsError('Existing demonstration retained')
    lines=['ffconcat version 1.0']
    for i,frame in enumerate(frames):
        path=(directory/'frames'/frame['file']).resolve().as_posix()
        seconds=max(0.1,(frames[i+1]['at_ms']-frame['at_ms'])/1000) if i+1<len(frames) else 4
        lines.extend([f"file '{path}'",f'duration {seconds:.6f}'])
    lines.append(lines[-2])
    inputs=directory/'frames.ffconcat';inputs.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','warning','-y','-f','concat','-safe','0',
             '-i',str(inputs),'-vf','scale=1180:1230:force_original_aspect_ratio=decrease,pad=1180:1230:(ow-iw)/2:(oh-ih)/2,setsar=1',
             '-r','15','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(target)]
    run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (directory/'ffmpeg.log').write_text(run.stderr,encoding='utf-8')
    if run.returncode:raise RuntimeError(f'Video encoding failed: {run.returncode}')
    decode=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(target),'-f','null','-'],capture_output=True)
    if decode.returncode:raise RuntimeError('Encoded demonstration did not decode')
    reader=imageio_ffmpeg.read_frames(str(target));metadata=next(reader);reader.close()
    save_json(directory/'video-info.json',metadata)
    (directory/'录像说明.txt').write_text('本视频由新版网页真实操作画面组成：选择图片、点击开始、实际加载两路模型并完成推理。按原始截图时间编码，保留结果阅读停顿，没有更改模型输出、分数或耗时。原始采集记录在recording.json，实际服务返回在live-run.json。编码采用CPU两线程。',encoding='utf-8')
    print('DEMO SAVED',target,metadata,flush=True)
