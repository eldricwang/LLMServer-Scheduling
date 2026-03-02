"""
First-Come-First-Serve Scheduler
"""

from typing import Optional
from .base import BaseScheduler
from ..core.request import Request
from ..core.state import SystemState


class FCFSScheduler(BaseScheduler):
    """First-Come-First-Serve scheduler"""
    
    name = "fcfs"
    
    def select_next(self, state: SystemState) -> Optional[Request]:
        if not state.waiting_queue:
            return None
        
        # Sort by arrival time
        sorted_queue = sorted(state.waiting_queue, key=lambda r: r.arrival_time)
        
        # Find first that fits
        for req in sorted_queue:
            if state.can_admit(req):
                return req
        return None
