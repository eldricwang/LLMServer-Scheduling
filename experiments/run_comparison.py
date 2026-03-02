#!/usr/bin/env python3
"""
Run comparison experiments across all schedulers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memsched import (
    get_model_config, SystemConfig,
    get_scheduler, SCHEDULERS,
    Simulator,
    WorkloadConfig, WorkloadGenerator,
    MetricsCollector,
)


def run_comparison(
    workload_type: str = "uniform",
    num_requests: int = 100,
    num_runs: int = 3,
):
    """Run comparison across all schedulers"""
    
    print("=" * 60)
    print("MemSched Comparison Experiment")
    print("=" * 60)
    
    # Configuration
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig(
        gpu_memory_mb=24 * 1024,
        model_memory_mb=14 * 1024,
        max_batch_size=32,
    )
    
    workload_config = WorkloadConfig(
        num_requests=num_requests,
        arrival_rate=10.0,
        seed=42,
    )
    
    print(f"\nModel: {model_config.name}")
    print(f"GPU Memory: {system_config.gpu_memory_mb / 1024:.0f} GB")
    print(f"Available KV Memory: {system_config.available_kv_memory_mb / 1024:.1f} GB")
    print(f"Num Requests: {num_requests}")
    print(f"Num Runs: {num_runs}")
    print(f"Workload Type: {workload_type}")
    
    # Metrics collector
    collector = MetricsCollector()
    
    # Schedulers to compare
    scheduler_names = ["fcfs", "sjf", "sjf_predicted", "mlfq", "vllm", "memsched"]
    
    print("\n" + "-" * 60)
    print("Running experiments...")
    print("-" * 60)
    
    for scheduler_name in scheduler_names:
        print(f"\n[{scheduler_name.upper()}]")
        
        for run_id in range(num_runs):
            # Generate workload (different seed each run)
            workload_config.seed = 42 + run_id
            generator = WorkloadGenerator(workload_config)
            
            if workload_type == "uniform":
                requests = generator.generate()
            elif workload_type == "bursty":
                requests = generator.generate_bursty()
            elif workload_type == "bimodal":
                requests = generator.generate_bimodal()
            else:
                requests = generator.generate()
            
            # Create scheduler and simulator
            scheduler = get_scheduler(scheduler_name)
            simulator = Simulator(model_config, system_config, scheduler)
            
            # Run simulation
            result = simulator.run(requests, show_progress=False)
            collector.add_result(scheduler_name, result)
            
            print(f"  Run {run_id + 1}: "
                  f"latency={result.avg_latency:.1f}ms, "
                  f"throughput={result.throughput:.2f} req/s")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summaries = collector.summarize_all(workload_type)
    
    # Header
    print(f"\n{'Scheduler':<15} {'Avg Latency':<18} {'P99 Latency':<18} "
          f"{'Throughput':<15} {'Preemptions':<12}")
    print("-" * 80)
    
    for summary in summaries:
        print(f"{summary.scheduler_name:<15} "
              f"{summary.avg_latency_mean:>8.1f}±{summary.avg_latency_std:<7.1f} "
              f"{summary.p99_latency_mean:>8.1f}±{summary.p99_latency_std:<7.1f} "
              f"{summary.throughput_mean:>7.2f}±{summary.throughput_std:<5.2f} "
              f"{summary.preemption_mean:>8.1f}")
    
    # Find best
    print("\n" + "-" * 60)
    best_latency = min(summaries, key=lambda s: s.avg_latency_mean)
    best_throughput = max(summaries, key=lambda s: s.throughput_mean)
    
    print(f"Best Avg Latency:  {best_latency.scheduler_name} ({best_latency.avg_latency_mean:.1f}ms)")
    print(f"Best Throughput:   {best_throughput.scheduler_name} ({best_throughput.throughput_mean:.2f} req/s)")
    
    # Improvement over FCFS
    fcfs_summary = next(s for s in summaries if s.scheduler_name == "fcfs")
    memsched_summary = next(s for s in summaries if s.scheduler_name == "memsched")
    
    latency_improvement = (fcfs_summary.avg_latency_mean - memsched_summary.avg_latency_mean) / fcfs_summary.avg_latency_mean * 100
    throughput_improvement = (memsched_summary.throughput_mean - fcfs_summary.throughput_mean) / fcfs_summary.throughput_mean * 100
    
    print(f"\nMemSched vs FCFS:")
    print(f"  Latency improvement: {latency_improvement:+.1f}%")
    print(f"  Throughput improvement: {throughput_improvement:+.1f}%")
    
    return collector


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run scheduler comparison")
    parser.add_argument("--workload", type=str, default="uniform",
                        choices=["uniform", "bursty", "bimodal"],
                        help="Workload type")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs per scheduler")
    
    args = parser.parse_args()
    
    run_comparison(
        workload_type=args.workload,
        num_requests=args.num_requests,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()
