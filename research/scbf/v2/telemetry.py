"""
SCBF v2 run telemetry — the CSV/JSON schemas proven across Ember III rounds 5-9.

RunLog keeps three streams: per-step timeseries rows, cadenced snapshot rows, and
a meta dict; flushes to timeseries.csv / snapshots.csv / meta.json. Rows are plain
dicts; schemas stay analysis-compatible with the existing ember3 tooling.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


class RunLog:
    def __init__(self, out_dir: str, meta: Optional[dict] = None):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.ts_rows: list = []
        self.snap_rows: list = []
        self.meta = dict(meta or {})
        self._t0 = time.time()

    def step(self, **row):
        self.ts_rows.append(row)

    def snapshot(self, **row):
        self.snap_rows.append(row)

    def finalize(self, **extra_meta) -> str:
        import pandas as pd
        if self.ts_rows:
            pd.DataFrame(self.ts_rows).to_csv(
                os.path.join(self.out_dir, "timeseries.csv"), index=False)
        if self.snap_rows:
            pd.DataFrame(self.snap_rows).to_csv(
                os.path.join(self.out_dir, "snapshots.csv"), index=False)
        self.meta.update(extra_meta)
        self.meta["duration_s"] = round(time.time() - self._t0, 1)
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        return self.out_dir
