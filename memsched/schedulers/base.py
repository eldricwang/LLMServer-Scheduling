"""
Base scheduler interface
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from ..core.request import Request
from ..core.state import SystemState


class BaseScheduler(ABC):
    """Abstract base class for schedulers"""
    
    name: str = "base"
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def select_next(self, state: SystemState) -> Optional[Request]:
        """Select next request to execute"""
        pass
    
    def should_preempt(
        self, state: SystemState, new_request: Request
    ) -> Optional[Request]:
        """Decide whether to preempt. Returns victim or None."""
        return None
    
    def on_request_arrival(self, state: SystemState, request: Request):
        pass
    
    def on_request_complete(self, state: SystemState, request: Request):
        pass
    
    def on_token_generated(self, state: SystemState, request: Request):
        pass
    
    def reset(self):
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
