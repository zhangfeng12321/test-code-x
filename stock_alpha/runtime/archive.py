from __future__ import annotations

from pathlib import Path
import shutil


def copy_if_exists(src: str | Path, dst: str | Path) -> Path | None:
    s = Path(src)
    if not s.exists():
        return None
    d = Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)
    return d
