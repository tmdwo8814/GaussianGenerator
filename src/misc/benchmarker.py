import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


class Benchmarker:
    def __init__(self, synchronize_device=None):
        self.execution_times = defaultdict(list)
        self.synchronize_device = synchronize_device

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        if self.synchronize_device is not None:
            torch.cuda.synchronize(self.synchronize_device)
        start_time = perf_counter()
        try:
            yield
        finally:
            if self.synchronize_device is not None:
                torch.cuda.synchronize(self.synchronize_device)
            end_time = perf_counter()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
