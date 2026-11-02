"""One official generic pretrained checkpoint, no retry and no GPU work."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask


if __name__ == '__main__':
    with ModelTask('download-swin'):
        url = 'https://download.pytorch.org/models/swin_t-704ceda3.pth'
        target = ROOT/'weights/swin_t-704ceda3.pth'
        if not target.exists():
            partial = target.with_suffix('.pth.part')
            command = ['curl.exe' if sys.platform=='win32' else 'curl','--fail','-L',
                       '--connect-timeout','15','--max-time','900','--retry','0',
                       '-o',str(partial),url]
            print('DOWNLOAD ONCE',url,flush=True)
            subprocess.run(command,check=True)
            with partial.open('r+b') as f:
                os.fsync(f.fileno())
            digest=hashlib.sha256(partial.read_bytes()).hexdigest()
            if not digest.startswith('704ceda3'):
                raise ValueError(f'Official hash prefix mismatch; partial retained: {digest}')
            partial.replace(target)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if not digest.startswith('704ceda3'):
            raise ValueError(f'Official hash prefix mismatch: {digest}')
        save_json(ROOT/'artifacts/optimization-v2/swin-download.json',{
            'source':url,'bytes':target.stat().st_size,'sha256':digest,
            'official_published_hash_prefix':'704ceda3','scope':'ImageNet generic initialization, not cat-finetuned weights'})
        print('VERIFIED',target.name,target.stat().st_size,digest,flush=True)
