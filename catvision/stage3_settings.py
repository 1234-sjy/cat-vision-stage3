"""Environment-based configuration shared by API and queue worker."""
from dataclasses import dataclass, field
import os
from pathlib import Path

from .common import ROOT


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///' + (ROOT/'runtime/stage3/database.db').as_posix()))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('CATVISION_DATA_DIR', str(ROOT/'runtime/stage3/jobs'))))
    jwt_secret: str = field(default_factory=lambda: os.getenv('JWT_SECRET', ''))
    amqp_url: str = field(default_factory=lambda: os.getenv('AMQP_URL', 'amqp://catvision:catvision@localhost:5672/catvision'))
    queue_name: str = 'catvision.inference'
    token_hours: int = 12
    max_pending: int = 8

    def prepare(self):
        if len(self.jwt_secret) < 32:
            raise ValueError('JWT_SECRET需要至少32个字符，请运行初始化配置脚本')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith('sqlite:///'):
            Path(self.database_url.removeprefix('sqlite:///')).parent.mkdir(parents=True, exist_ok=True)
