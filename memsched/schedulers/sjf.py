"""
Shortest Job First Schedulers
"""

from typing import Optional
from .base import BaseScheduler
from ..core.request import Request
from ..core.state import SystemState


class SJFScheduler(BaseScheduler):
    """Shortest Job First (by prompt length)"""
    
    name = "sjf"
    
    def select_next(self, state: SystemState) -> Optional[Request]:
        if not state.waiting_queue:
            return None
        
        sorted_queue = sorted(state.waiting_queue, key=lambda r: r.prompt_tokens)
        
        for req in sorted_queue:
            if state.can_admit(req):
                return req
        return None


class SJFPredictedScheduler(BaseScheduler):
    """Shortest Job First (by predicted total length)"""
    
    name = "sjf_predicted"
    
    def select_next(self, state: SystemState) -> Optional[Request]:
        if not state.waiting_queue:
            return None
        
        sorted_queue = sorted(state.waiting_queue, key=lambda r: r.estimated_total_tokens)
        
        for req in sorted_queue:
            if state.can_admit(req):
                return req
        return None
