#!/usr/bin/env python3
"""
Full evaluation for paper results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memsched import (
    get_model_config, SystemConfig,
    get_scheduler,
    Simulator,
    Request,
)
import random
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    scheduler: str
    scenario: str
    avg_latency: float
    p50_latency: float
    p99_latency: float
    throughput: float
    memory_util: float
    completed: int
    total: int


def create_workload(
    scenario: str,
    num_requests: int = 150,
    seed: int = 42
) -> List[Request]:
    """Create workload for different scenarios"""
    random.seed(seed)
    requests = []
    current_time = 0.0
    
    for i in range(num_requests):
        if scenario == "uniform":
            prompt = random.randint(100, 500)
            output = random.randint(50, 250)
            arrival_rate = 15.0
        
        elif scenario == "high_load":
            prompt = random.randint(200, 800)
            output = random.randint(100, 400)
            arrival_rate = 30.0
        
        elif scenario == "bimodal":
            if random.random() < 0.5:
                prompt = random.randint(30, 100)
                output = random.randint(20, 60)
            else:
                prompt = random.randint(500, 1000)
                output = random.randint(300, 600)
            arrival_rate = 20.0
        
        elif scenario == "bursty":
            prompt = random.randint(150, 600)
            output = random.randint(80, 300)
            # Bursty arrivals
            if random.random() < 0.3:
                arrival_rate = 100.0  # Burst
            else:
                arrival_rate = 10.0
        
        elif scenario == "long_sequences":
            prompt = random.randint(400, 1200)
            output = random.randint(200, 600)
            arrival_rate = 10.0
        
        else:
            prompt = random.randint(100, 500)
            output = random.randint(50, 250)
            arrival_rate = 15.0
        
        inter_arrival = random.expovariate(arrival_rate) * 1000
        current_time += inter_arrival
        
        priority = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]
        
        requests.append(Request(
            id=i,
            arrival_time=current_time,
            prompt_tokens=prompt,
            max_output_tokens=output + 50,
            actual_output_tokens=output,
            priority=priority,
        ))
    
    return requests


def run_experiment(
    scenario: str,
    memory_gb: float,
    num_requests: int = 150,
    num_runs: int = 3,
) -> Dict[str, List[ExperimentResult]]:
    """Run experiment for all schedulers"""
    
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig(
        gpu_memory_mb=int(memory_gb * 1024),
        model_memory_mb=14 * 1024,
        max_batch_size=16,
        prefill_time_per_token_ms=0.3,
        decode_time_per_token_ms=5.0,
    )
    
    schedulers = ["fcfs", "sjf", "sjf_predicted", "mlfq", "memsched"]
    results = {s: [] for s in schedulers}
    
    for run in range(num_runs):
        seed = 42 + run * 100
        workload = create_workload(scenario, num_requests, seed)
        
        for sched_name in schedulers:
            # Copy workload
            test_workload = [
                Request(
                    id=r.id,
                    arrival_time=r.arrival_time,
                    prompt_tokens=r.prompt_tokens,
                    max_output_tokens=r.max_output_tokens,
                    actual_output_tokens=r.actual_output_tokens,
                    priority=r.priority,
                )
                for r in workload
            ]
            
            scheduler = get_scheduler(sched_name)
            simulator = Simulator(model_config, system_config, scheduler)
            result = simulator.run(test_workload, show_progress=False)
            
            results[sched_name].append(ExperimentResult(
                scheduler=sched_name,
                scenario=scenario,
                avg_latency=result.avg_latency,
                p50_latency=result.p50_latency,
                p99_latency=result.p99_latency,
                throughput=result.throughput,
                memory_util=result.avg_memory_utilization,
                completed=len(result.completed_requests),
                total=num_requests,
            ))
    
    return results


def aggregate_results(results: Dict[str, List[ExperimentResult]]) -> Dict[str, dict]:
    """Aggregate results across runs"""
    aggregated = {}
    
    for sched, runs in results.items():
        n = len(runs)
        aggregated[sched] = {
            "avg_latency": sum(r.avg_latency for r in runs) / n,
            "p50_latency": sum(r.p50_latency for r in runs) / n,
            "p99_latency": sum(r.p99_latency for r in runs) / n,
            "throughput": sum(r.throughput for r in runs) / n,
            "memory_util": sum(r.memory_util for r in runs) / n,
            "completion_rate": sum(r.completed / r.total for r in runs) / n,
        }
    
    return aggregated


def print_table(title: str, results: Dict[str, dict], metrics: List[str]):
    """Print formatted table"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    # Header
    header = f"{'Scheduler':<15}"
    for m in metrics:
        header += f"{m:<12}"
    print(header)
    print("-" * 70)
    
    # Data
    for sched, data in results.items():
        row = f"{sched:<15}"
        for m in metrics:
            val = data.get(m.lower().replace(" ", "_"), 0)
            if "latency" in m.lower():
                row += f"{val:>8.0f}ms  "
            elif "throughput" in m.lower():
                row += f"{val:>8.2f}    "
            elif "rate" in m.lower() or "util" in m.lower():
                row += f"{val:>8.1%}    "
            else:
                row += f"{val:>10.2f}  "
        print(row)


def main():
    print("=" * 70)
    print("MEMSCHED FULL EVALUATION")
    print("=" * 70)
    
    scenarios = [
        ("uniform", 16.0, "Uniform Workload (16GB GPU)"),
        ("high_load", 15.0, "High Load (15GB GPU)"),
        ("bimodal", 16.0, "Bimodal Workload"),
        ("bursty", 16.0, "Bursty Arrivals"),
        ("long_sequences", 15.0, "Long Sequences"),
    ]
    
    all_results = {}
    
    for scenario, memory_gb, description in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {description}")
        print(f"Memory: {memory_gb}GB, Available KV: {(memory_gb-14)*1024:.0f}MB")
        print(f"{'='*70}")
        
        results = run_experiment(scenario, memory_gb, num_requests=150, num_runs=3)
        aggregated = aggregate_results(results)
        all_results[scenario] = aggregated
        
        print_table(
            f"Results: {description}",
            aggregated,
            ["Avg Latency", "P99 Latency", "Throughput"]
        )
        
        # Calculate improvements
        fcfs = aggregated["fcfs"]
        print(f"\nImprovement over FCFS:")
        for sched, data in aggregated.items():
            if sched == "fcfs":
                continue
            lat_imp = (fcfs["avg_latency"] - data["avg_latency"]) / fcfs["avg_latency"] * 100
            thr_imp = (data["throughput"] - fcfs["throughput"]) / fcfs["throughput"] * 100
            print(f"  {sched:<15}: Latency {lat_imp:+.1f}%, Throughput {thr_imp:+.1f}%")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    # Average improvement across all scenarios
    print("\nAverage improvement over FCFS across all scenarios:")
    
    schedulers = ["sjf", "sjf_predicted", "mlfq", "memsched"]
    
    for sched in schedulers:
        lat_improvements = []
        thr_improvements = []
        
        for scenario, results in all_results.items():
            fcfs = results["fcfs"]
            data = results[sched]
            lat_improvements.append(
                (fcfs["avg_latency"] - data["avg_latency"]) / fcfs["avg_latency"] * 100
            )
            thr_improvements.append(
                (data["throughput"] - fcfs["throughput"]) / fcfs["throughput"] * 100
            )
        
        avg_lat = sum(lat_improvements) / len(lat_improvements)
        avg_thr = sum(thr_improvements) / len(thr_improvements)
        
        print(f"  {sched:<15}: Latency {avg_lat:+.1f}%, Throughput {avg_thr:+.1f}%")
    
    # Best scheduler per scenario
    print("\nBest scheduler per scenario:")
    for scenario, results in all_results.items():
        best_lat = min(results.items(), key=lambda x: x[1]["avg_latency"])
        best_thr = max(results.items(), key=lambda x: x[1]["throughput"])
        print(f"  {scenario:<15}: Best Latency={best_lat[0]}, Best Throughput={best_thr[0]}")


if __name__ == "__main__":
    main()
