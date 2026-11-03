"""Create deployment secrets locally; never overwrite an existing configuration."""
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]
target = ROOT/'.env'
if target.exists():
    print('已有.env，保留当前配置。')
else:
    target.write_text('DB_PASSWORD='+secrets.token_hex(24)+'\nMQ_PASSWORD='+secrets.token_hex(24)+
        '\nJWT_SECRET='+secrets.token_hex(48)+'\nCATVISION_MODEL_ROOT=./weights\nBIND_ADDRESS=127.0.0.1\nWEB_PORT=8080\n',encoding='utf-8')
    try:
        target.chmod(0o600)
    except OSError:
        pass
    print('已创建.env，包含随机服务密码和JWT密钥；不要上传此文件。')
for name in ['data/jobs','model-runtime','model-logs']:
    (ROOT/'runtime/stage3'/name).mkdir(parents=True,exist_ok=True)
