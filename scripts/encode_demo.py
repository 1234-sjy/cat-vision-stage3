"""Encode actual browser screenshots; preserve capture timing and model displays."""
import json
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT,save_json


if __name__=='__main__':
    import imageio_ffmpeg
    from PIL import Image
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/demo-v2')
    parser.add_argument('--fps',type=int,default=15)
    parser.add_argument('--trim-finished-idle',action='store_true')
    args=parser.parse_args()
    directory=args.directory.resolve()
    recording=json.loads((directory/'recording.json').read_text())
    frames=recording['frames']
    target=directory/'实际运行演示.mp4'
    if target.exists():raise FileExistsError('Existing demonstration retained')
    lines=['ffconcat version 1.0']
    widths=[];heights=[]
    for i,frame in enumerate(frames):
        path=(directory/'frames'/frame['file']).resolve().as_posix()
        with Image.open(path) as source:
            widths.append(source.width);heights.append(source.height)
        seconds=max(0.1,(frames[i+1]['at_ms']-frame['at_ms'])/1000) if i+1<len(frames) else 4
        # Never shorten inference; only an idle capture gap after the result is complete.
        if args.trim_finished_idle and frame['at_ms']>=recording['inference_completed_at_ms'] and seconds>1 and i+1<len(frames):
            seconds=0.3
        # Preserve sub-100ms cadence for continuous video recordings.
        if i+1<len(frames) and (frames[i+1]['at_ms']-frame['at_ms'])<100:
            seconds=max(0.001,(frames[i+1]['at_ms']-frame['at_ms'])/1000)
        lines.extend([f"file '{path}'",f'duration {seconds:.6f}'])
    lines.append(lines[-2])
    inputs=directory/'frames.ffconcat';inputs.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    width=(max(widths)+1)//2*2; height=(max(heights)+1)//2*2
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','warning','-y','-f','concat','-safe','0',
             '-i',str(inputs),'-vf',f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1',
             '-r',str(args.fps),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(target)]
    run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (directory/'ffmpeg.log').write_text(run.stderr,encoding='utf-8')
    if run.returncode:raise RuntimeError(f'Video encoding failed: {run.returncode}')
    decode=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(target),'-f','null','-'],capture_output=True)
    if decode.returncode:raise RuntimeError('Encoded demonstration did not decode')
    reader=imageio_ffmpeg.read_frames(str(target));metadata=next(reader);reader.close()
    save_json(directory/'video-info.json',metadata)
    note='本视频来自网页真实上传和重新推理。快速模式在录制前预加载模型，取消人为推理暂停；计算仍依次执行，不复用答案。连续采集画面，输出30fps；识别等待保留真实速度，结果完成后的多余静止间隔适当压缩。模型输出、分数和界面耗时均未修改。recording.json保留原始帧时间，live-run.json保留服务返回。'
    if not args.trim_finished_idle:
        note='本视频连续采集实际网页的选图、点击识别和两路输出，按原始画面时间编码。识别过程完整保留，没有剪掉等待、加速推理或修改分数及耗时。原始帧时间在recording.json，服务返回在live-run.json。'
    (directory/'录像说明.txt').write_text(note,encoding='utf-8')
    print('DEMO SAVED',target,metadata,flush=True)
