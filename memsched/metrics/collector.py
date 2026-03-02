"""
Metrics Collection and Analysis
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricsSummary:
    """Summary of metrics across multiple runs"""
    
    scheduler_name: str
    workload_name: str
    num_runs: int
    
    # Latency metrics
    avg_latency_mean: float = 0.0
    avg_latency_std: float = 0.0
    p50_latency_mean: float = 0.0
    p50_latency_std: float = 0.0
    p99_latency_mean: float = 0.0
    p99_latency_std: float = 0.0
    
    # Throughput metrics
    throughput_mean: float = 0.0
    throughput_std: float = 0.0
    token_throughput_mean: float = 0.0
    token_throughput_std: float = 0.0
    
    # Other metrics
    preemption_mean: float = 0.0
    memory_util_mean: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheduler": self.scheduler_name,
            "workload": self.workload_name,
            "num_runs": self.num_runs,
            "avg_latency": f"{self.avg_latency_mean:.2f}±{self.avg_latency_std:.2f}",
            "p99_latency": f"{self.p99_latency_mean:.2f}±{self.p99_latency_std:.2f}",
            "throughput": f"{self.throughput_mean:.2f}±{self.throughput_std:.2f}",
        }


class MetricsCollector:
    """Collect and aggregate metrics from simulation results"""
    
    def __init__(self):
        self.results: Dict[str, List] = {}
    
    def add_result(self, scheduler_name: str, result):
        """Add a simulation result"""
        if scheduler_name not in self.results:
            self.results[scheduler_name] = []
        self.results[scheduler_name].append(result)
    
    def clear(self):
        """Clear all results"""
        self.results.clear()
    
    def summarize(self, scheduler_name: str, workload_name: str = "default") -> MetricsSummary:
        """Summarize results for a scheduler"""
        if scheduler_name not in self.results:
            raise ValueError(f"No results for scheduler: {scheduler_name}")
        
        results = self.results[scheduler_name]
        n = len(results)
        
        avg_latencies = [r.avg_latency for r in results]
        p50_latencies = [r.p50_latency for r in results]
        p99_latencies = [r.p99_latency for r in results]
        throughputs = [r.throughput for r in results]
        token_throughputs = [r.token_throughput for r in results]
        preemptions = [r.preemption_count for r in results]
        memory_utils = [r.avg_memory_utilization for r in results]
        
        return MetricsSummary(
            scheduler_name=scheduler_name,
            workload_name=workload_name,
            num_runs=n,
            avg_latency_mean=float(np.mean(avg_latencies)),
            avg_latency_std=float(np.std(avg_latencies)),
            p50_latency_mean=float(np.mean(p50_latencies)),
            p50_latency_std=float(np.std(p50_latencies)),
            p99_latency_mean=float(np.mean(p99_latencies)),
            p99_latency_std=float(np.std(p99_latencies)),
            throughput_mean=float(np.mean(throughputs)),
            throughput_std=float(np.std(throughputs)),
            token_throughput_mean=float(np.mean(token_throughputs)),
            token_throughput_std=float(np.std(token_throughputs)),
            preemption_mean=float(np.mean(preemptions)),
            memory_util_mean=float(np.mean(memory_utils)),
        )
    
    def summarize_all(self, workload_name: str = "default") -> List[MetricsSummary]:
        """Summarize results for all schedulers"""
        summaries = []
        for scheduler_name in self.results:
            summaries.append(self.summarize(scheduler_name, workload_name))
        return summaries


def compute_slo_metrics(requests: List, latency_slo_ms: float = 1000.0) -> Dict[str, float]:
    """Compute SLO-related metrics"""
    if not requests:
        return {}
    
    latencies = [r.latency for r in requests]
    slo_violations = sum(1 for l in latencies if l > latency_slo_ms)
    
    return {
        "slo_attainment": 1 - slo_violations / len(requests),
        "slo_violations": slo_violations,
    }
