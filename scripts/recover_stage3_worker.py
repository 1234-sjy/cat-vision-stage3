"""Operator acknowledgement after inspecting a failed model log; never auto-run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT
from catvision.run_control import GpuReservation

parser = argparse.ArgumentParser()
parser.add_argument('--acknowledge',action='store_true',help='Confirm that the prior model error log has been reviewed')
args = parser.parse_args()
state_path=ROOT/'runtime/model-task-state.json'
if not state_path.exists():
    print('没有待处理的模型状态。')
    raise SystemExit(0)
record=json.loads(state_path.read_text(encoding='utf-8'))
print('上次状态：',record.get('state'),'；原始日志：',record.get('logs'))
if not args.acknowledge:
    print('先查看日志。确认恢复时传入--acknowledge；此脚本不会启动推理。')
    raise SystemExit(0)
with GpuReservation(ROOT):
    if record.get('state') in {'running','failed','interrupted'}:
        folder=ROOT/'runtime/acknowledged-states'
        folder.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (folder/f'{stamp}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        state_path.write_text(json.dumps({'state':'acknowledged','previous_log':record.get('logs'),
            'acknowledged_utc':datetime.now(timezone.utc).isoformat(),'automatic_restart':False},ensure_ascii=False,indent=2),encoding='utf-8')
        print('已保留旧状态并确认处理；请手动启动工作端。')
    else:
        print('当前不需要恢复。')
