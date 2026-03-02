#!/usr/bin/env python3
"""
Run a single experiment with specified scheduler
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memsched import (
    get_model_config, SystemConfig,
    get_scheduler, SCHEDULERS,
    Simulator,
    WorkloadConfig, WorkloadGenerator,
)


def run_single(
    scheduler_name: str = "fcfs",
    num_requests: int = 50,
    arrival_rate: float = 10.0,
    seed: int = 42,
):
    """Run single experiment"""
    
    print("=" * 50)
    print(f"Running: {scheduler_name.upper()}")
    print("=" * 50)
    
    # Configuration
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig(
        gpu_memory_mb=24 * 1024,
        model_memory_mb=14 * 1024,
        max_batch_size=32,
    )
    
    workload_config = WorkloadConfig(
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        seed=seed,
    )
    
    # Generate workload
    generator = WorkloadGenerator(workload_config)
    requests = generator.generate()
    
    print(f"\nWorkload Statistics:")
    print(f"  Num requests: {len(requests)}")
    print(f"  Avg prompt tokens: {sum(r.prompt_tokens for r in requests) / len(requests):.1f}")
    print(f"  Avg output tokens: {sum(r.actual_output_tokens for r in requests) / len(requests):.1f}")
    
    # Create scheduler and simulator
    scheduler = get_scheduler(scheduler_name)
    simulator = Simulator(model_config, system_config, scheduler)
    
    # Run simulation
    print(f"\nRunning simulation...")
    result = simulator.run(requests, show_progress=True)
    
    # Print results
    print(f"\nResults:")
    print(f"  Total time: {result.total_time_ms:.1f} ms ({result.total_time_ms/1000:.2f} s)")
    print(f"  Completed: {len(result.completed_requests)} requests")
    print(f"  Avg latency: {result.avg_latency:.1f} ms")
    print(f"  P50 latency: {result.p50_latency:.1f} ms")
    print(f"  P99 latency: {result.p99_latency:.1f} ms")
    print(f"  Throughput: {result.throughput:.2f} req/s")
    print(f"  Token throughput: {result.token_throughput:.1f} tok/s")
    print(f"  Preemptions: {result.preemption_count}")
    print(f"  Avg memory util: {result.avg_memory_utilization:.1%}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run single experiment")
    parser.add_argument("--scheduler", type=str, default="fcfs",
                        choices=list(SCHEDULERS.keys()),
                        help="Scheduler to use")
    parser.add_argument("--num-requests", type=int, default=50,
                        help="Number of requests")
    parser.add_argument("--arrival-rate", type=float, default=10.0,
                        help="Arrival rate (requests/second)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    run_single(
        scheduler_name=args.scheduler,
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
