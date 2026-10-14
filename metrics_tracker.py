import json 
from pathlib import Path
from dataclasses import asdict
from datetime import datetime


class MetricTracker():
    def __init__(self, runs_root, config, run_name = ""):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(runs_root) / (f"{stamp}_{run_name}" if run_name else stamp)
        self.run_dir.mkdir(parents=True)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        self.file = open(self.run_dir / "metrics.jsonl", "w")

    def log(self, step, metrics):
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()