from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
BREEDS = {
    "Abyssinian": "阿比西尼亚猫", "Bengal": "孟加拉猫",
    "Birman": "伯曼猫", "Bombay": "孟买猫",
    "British_Shorthair": "英国短毛猫", "Egyptian_Mau": "埃及猫",
    "Maine_Coon": "缅因猫", "Persian": "波斯猫", "Ragdoll": "布偶猫",
    "Russian_Blue": "俄罗斯蓝猫", "Siamese": "暹罗猫", "Sphynx": "斯芬克斯猫",
    "American_Bobtail": "美国短尾猫", "American_Curl": "美国卷耳猫",
    "American_Shorthair": "美国短毛猫", "Exotic_Shorthair": "异国短毛猫",
    "Manx": "曼岛猫", "Norwegian_Forest": "挪威森林猫",
    "Scottish_Fold": "苏格兰折耳猫", "Turkish_Angora": "土耳其安哥拉猫",
}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_image(source: bytes | str | Path, max_side: int | None = None) -> Image.Image:
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as raw:
            if raw.width * raw.height > 40_000_000:
                raise ValueError("图片超过4000万像素，请缩小后重试")
            image = ImageOps.exif_transpose(raw)
            if "A" in image.getbands() or image.mode == "P":
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(white, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取这张图片，请使用JPG、PNG或WebP文件") from exc
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def image_hash(image: Image.Image) -> str:
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def parse_vlm_output(text: str) -> dict:
    error = {"status": "parse_error", "has_cat": None, "breed": None,
             "breed_zh": None, "features": [], "raw_text": text}
    decoder = json.JSONDecoder()
    data = None
    for index, char in enumerate(text):
        if char == "{":
            try:
                candidate, _ = decoder.raw_decode(text[index:])
                if isinstance(candidate, dict) and "has_cat" in candidate:
                    data = candidate
                    break
            except json.JSONDecodeError:
                continue
    if not data or not {"has_cat", "breed", "features"}.issubset(data) or type(data.get("has_cat")) is not bool:
        return error
    if not isinstance(data["features"], list) or any(not isinstance(v, str) for v in data["features"]):
        return error
    if not data["has_cat"]:
        return {**error, "status": "no_cat", "has_cat": False}
    aliases = {k.replace("_", " ").casefold(): k for k in BREEDS}
    aliases.update({v: k for k, v in BREEDS.items()})
    breed = data.get("breed")
    if not isinstance(breed, str):
        return error
    if breed.strip().casefold() in {"other", "unknown", "其他品种"}:
        return {**error, "status": "unsupported", "has_cat": True, "breed_zh": "其他品种"}
    canonical = aliases.get(breed.strip().replace("_", " ").casefold())
    features = data.get("features", [])
    if not canonical or not isinstance(features, list) or any(not isinstance(v, str) for v in features):
        return error
    return {"status": "ok", "has_cat": True, "breed": canonical,
            "breed_zh": BREEDS[canonical], "features": [v[:160] for v in features[:3]],
            "raw_text": text}
