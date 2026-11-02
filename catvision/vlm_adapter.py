"""Portable adapter paths shared by loading, readiness and result provenance."""
from pathlib import Path


def resolve_adapter(root: Path, settings: dict) -> Path | None:
    value = settings.get('adapter')
    if value is None:
        return None
    if (not isinstance(value, str) or not value.strip() or ':' in value
            or value.startswith(('/', '\\'))
            or '..' in value.replace('\\', '/').split('/')):
        raise ValueError('Adapter must use a project-relative directory')
    root = Path(root).resolve()
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Adapter path must stay inside the project')
    return path


def adapter_files(path: Path) -> list[Path]:
    return [path/'adapter_config.json', path/'adapter_model.safetensors']
