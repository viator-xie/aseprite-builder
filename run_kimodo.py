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


def copy_result_file(value, dst_dir: Path) -> list[str]:
    """Collect any local file paths returned by gradio_client."""
    copied: list[str] = []
    if value is None:
        return copied
    if isinstance(value, (list, tuple)):
        for item in value:
            copied.extend(copy_result_file(item, dst_dir))
        return copied
    if isinstance(value, dict):
        for item in value.values():
            copied.extend(copy_result_file(item, dst_dir))
        return copied

    # gradio_client may return a pathlib-like object or a FileData-ish value.
    candidates = []
    if isinstance(value, (str, os.PathLike)):
        candidates.append(Path(value))
    for attr in ("path", "name"):
        p = getattr(value, attr, None)
        if p:
            candidates.append(Path(str(p)))

    for src in candidates:
        if src.is_file():
            dst = dst_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def main() -> int:
    print(f"Connecting to Hugging Face Space: {SPACE}")
    token = os.environ.get("HF_TOKEN") or None
    kwargs = {}
    if token:
        # Current gradio_client uses hf_token; keeping this conditional means
        # the workflow also works anonymously when no secret is configured.
        kwargs["hf_token"] = token

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

    print("\n=== Kimodo request ===")
    print(json.dumps({
        "prompt": PROMPT,
        "model": MODEL,
        "character": CHARACTER,
        "duration": DURATION,
        "seed": SEED,
    }, indent=2, ensure_ascii=False))

    # /generate is the Space's stable simple text-to-motion Gradio endpoint.
    # It returns (model_file, status_text, rewritten_prompt).
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

    copied = copy_result_file(result, OUT)
    print("\n=== Copied output files ===")
    for p in copied:
        print(p)

    if not any(Path(p).suffix.lower() in {".fbx", ".glb", ".gltf"} for p in copied):
        print("ERROR: generation returned no FBX/GLB file", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
