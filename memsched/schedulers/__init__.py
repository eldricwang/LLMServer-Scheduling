from .base import BaseScheduler
from .fcfs import FCFSScheduler
from .sjf import SJFScheduler, SJFPredictedScheduler
from .mlfq import MLFQScheduler
from .vllm_sched import VLLMScheduler
from .memsched import MemSchedScheduler

SCHEDULERS = {
    "fcfs": FCFSScheduler,
    "sjf": SJFScheduler,
    "sjf_predicted": SJFPredictedScheduler,
    "mlfq": MLFQScheduler,
    "vllm": VLLMScheduler,
    "memsched": MemSchedScheduler,
}

def get_scheduler(name: str, **kwargs) -> BaseScheduler:
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(SCHEDULERS.keys())}")
    return SCHEDULERS[name](**kwargs)

__all__ = [
    "BaseScheduler",
    "FCFSScheduler", "SJFScheduler", "SJFPredictedScheduler",
    "MLFQScheduler", "VLLMScheduler", "MemSchedScheduler",
    "SCHEDULERS", "get_scheduler",
]
