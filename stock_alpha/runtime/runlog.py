from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import uuid


@dataclass
class RunLogger:
    root: Path | str = Path("runs")
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"

    def event(self, step: str, status: str, **payload) -> None:
        row = {"ts": datetime.now().isoformat(timespec="seconds"), "run_id": self.run_id, "step": step, "status": status, **payload}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{self.run_id}] {step}: {status}")

    def write_summary(self, **payload) -> Path:
        p = self.dir / "summary.json"
        p.write_text(json.dumps({"run_id": self.run_id, **payload}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p
