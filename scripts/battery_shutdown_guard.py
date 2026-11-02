"""One-session Windows battery guard explicitly armed by the user.

No startup task, clock changes, GPU queries, model retries or forced app close.
At 12% request a training checkpoint; at 10% issue the Windows shutdown command.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


class SystemPowerStatus(ctypes.Structure):
    _fields_ = [('ACLineStatus', wintypes.BYTE), ('BatteryFlag', wintypes.BYTE),
        ('BatteryLifePercent', wintypes.BYTE), ('SystemStatusFlag', wintypes.BYTE),
        ('BatteryLifeTime', wintypes.DWORD), ('BatteryFullLifeTime', wintypes.DWORD)]


def read_battery():
    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise ctypes.WinError()
    return {'battery_percent': None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent),
            'ac_online': status.ACLineStatus == 1}


def durable_write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(args):
    if not args.arm:
        raise RuntimeError('Must be explicitly armed for this user-requested shutdown')
    # Keep training alive when idle; display sleep and manual shutdown remain available.
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError()
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output/'state.json'
    state = {'state': 'monitoring', 'pid': os.getpid(), 'shutdown_percent': 10,
             'checkpoint_percent': 12, 'stop_file': str(args.stop_file.resolve()),
             'user_authorized': True, 'automatic_startup': False, 'prevent_idle_system_sleep': True,
             'created_utc': datetime.now(timezone.utc).isoformat()}
    durable_write(state_path, state)
    with (args.output/'battery.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        while True:
            power = read_battery()
            row = {'time_utc': datetime.now(timezone.utc).isoformat(), **power}
            log.write(json.dumps(row)+'\n')
            log.flush()
            os.fsync(log.fileno())
            percent = power['battery_percent']
            if percent is not None and percent <= 12 and not args.stop_file.exists():
                args.stop_file.parent.mkdir(parents=True, exist_ok=True)
                durable_write(args.stop_file, {**row, 'reason': 'checkpoint_before_user_requested_10_percent_shutdown'})
                state['checkpoint_requested_utc'] = row['time_utc']
            state.update(last_reading=row)
            if percent is not None and percent <= 10:
                state.update(state='shutdown_requested', shutdown_requested_utc=row['time_utc'])
                durable_write(state_path, state)
                subprocess.run(['shutdown.exe', '/s', '/t', '0', '/d', 'p:0:0',
                    '/c', '按用户要求：电量已降到10%，保存的训练检查点可下次继续使用。'],
                    check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                return
            durable_write(state_path, state)
            time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
