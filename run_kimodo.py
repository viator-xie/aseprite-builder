from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from gradio_client import Client

SPACE = os.environ.get("ANIMOFLOW_SPACE", "AnimoFlow/animoflow-demo")
PROMPT = os.environ.get(
    "KIMODO_PROMPT",
    "A timid young woman standing idle nervously with both arms hanging naturally at her sides, palms facing inward toward the thighs, thumbs pointing forward, shoulders slightly hunched and inward, head slightly lowered, subtle nervous breathing and tiny cautious weight shifts, realistic restrained body language, no weapon, no magic, no exaggerated gestures.",
)
MODEL = os.environ.get("KIMODO_MODEL", "kimodo")
CHARACTER = os.environ.get("KIMODO_CHARACTER", "Y_bot")
DURATION = float(os.environ.get("KIMODO_DURATION", "5.0"))
SEED = int(os.environ.get("KIMODO_SEED", "42"))
OUT = Path(os.environ.get("KIMODO_OUT", "artifacts"))
OUT.mkdir(parents=True, exist_ok=True)


def copy_generated_file(result) -> Path | None:
    """Copy only the first Gradio return value, which is the generated model file."""
    if not isinstance(result, (list, tuple)) or not result:
        return None
    value = result[0]
    candidates: list[Path] = []
    if isinstance(value, (str, os.PathLike)):
        candidates.append(Path(value))
    for attr in ("path", "name"):
        p = getattr(value, attr, None)
        if p:
            candidates.append(Path(str(p)))
    for src in candidates:
        try:
            if src.is_file():
                dst = OUT / src.name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                return dst
        except OSError:
            continue
    return None


def main() -> int:
    print(f"Connecting to Hugging Face Space: {SPACE}")
    token = os.environ.get("HF_TOKEN") or None
    kwargs = {"hf_token": token} if token else {}
    client = Client(SPACE, **kwargs)

    print("\n=== Space API ===")
    try:
        api_info = client.view_api(return_format="dict")
        print(json.dumps(api_info, indent=2, ensure_ascii=False, default=str))
        (OUT / "api_info.json").write_text(
            json.dumps(api_info, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"view_api failed (non-fatal): {exc}")

    request_meta = {
        "prompt": PROMPT,
        "model": MODEL,
        "character": CHARACTER,
        "duration": DURATION,
        "seed": SEED,
    }
    print("\n=== Kimodo request ===")
    print(json.dumps(request_meta, indent=2, ensure_ascii=False))
    (OUT / "request.json").write_text(
        json.dumps(request_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    result = client.predict(
        PROMPT,
        MODEL,
        CHARACTER,
        DURATION,
        SEED,
        api_name="/generate",
    )

    print("\n=== Raw result ===")
    print(repr(result))
    (OUT / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    generated = copy_generated_file(result)
    if generated is None or generated.suffix.lower() not in {".fbx", ".glb", ".gltf"}:
        print("ERROR: generation returned no FBX/GLB file", file=sys.stderr)
        return 2

    print(f"Generated file copied to: {generated}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
