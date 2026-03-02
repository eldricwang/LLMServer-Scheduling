好的，我重新写一个更清晰的版本：

MemSched: Memory-Aware Scheduling for LLM Serving
A simulation framework for evaluating KV cache-aware scheduling algorithms in Large Language Model (LLM) serving systems.

Overview
MemSched provides a discrete-event simulator to study how different scheduling strategies affect LLM serving performance under memory constraints. The framework is designed to be extensible and can serve as a foundation for implementing real-world LLM serving schedulers.

System Architecture
                              ┌─────────────────┐
                              │  User Requests  │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           REQUEST GENERATOR                                   │
│                                                                              │
│   Generates synthetic workloads with configurable patterns:                  │
│   • Uniform: Constant arrival rate                                           │
│   • Bimodal: Mix of short and long requests                                  │
│   • Bursty: Variable arrival rate with spikes                                │
│                                                                              │
│   Output: Request(prompt_len, output_len, arrival_time)                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ requests
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SCHEDULER                                        │
│                                                                              │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐           │
│   │  FCFS   │ │   SJF   │ │  MLFQ   │ │  vLLM   │ │  MemSched   │           │
│   │         │ │         │ │         │ │  Style  │ │  (Ours)     │           │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────┘           │
│                                                                              │
│   Input:  Pending requests + Available memory                                │
│   Output: Batch of requests to process                                       │
│                                                                              │
│   Responsibilities:                                                          │
│   • Maintain waiting queue                                                   │
│   • Select requests for next batch                                           │
│   • Handle preemption decisions                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ batch
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              LLM ENGINE                                       │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                       KV CACHE MANAGER                              │    │
│   │                                                                     │    │
│   │   • allocate(request) ──→ Reserve memory for new request           │    │
│   │   • extend(request)   ──→ Grow cache as tokens generated           │    │
│   │   • evict(request)    ──→ Free memory when preempted               │    │
│   │   • get_usage()       ──→ Report current memory utilization        │    │
│   │                                                                     │    │
│   │   Memory Model: Each token requires (num_layers × 2 × hidden_size) │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                       │                                      │
│                                       │ memory status                        │
│                                       ▼                                      │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                       TOKEN GENERATOR                               │    │
│   │                                                                     │    │
│   │   Prefill Phase:                                                    │    │
│   │   • Process all prompt tokens in parallel                          │    │
│   │   • Time = prompt_len × prefill_time_per_token                     │    │
│   │                                                                     │    │
│   │   Decode Phase:                                                     │    │
│   │   • Generate one token per iteration                               │    │
│   │   • Time = decode_time_per_token (batched)                         │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ completed requests
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           METRICS COLLECTOR                                   │
│                                                                              │
│   Latency Metrics:                                                           │
│   • Time-To-First-Token (TTFT)                                               │
│   • Time-Per-Output-Token (TPOT)                                             │
│   • End-to-End Latency                                                       │
│   • P50 / P99 Latency                                                        │
│                                                                              │
│   Throughput Metrics:                                                        │
│   • Requests per second                                                      │
│   • Tokens per second                                                        │
│                                                                              │
│   Resource Metrics:                                                          │
│   • Memory utilization over time                                             │
│   • Preemption count                                                         │
│   • Batch size distribution                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
Data Flow

1. Request Arrival
   ┌─────────────────────────────────────────────────────────────┐
   │ Request {                                                    │
   │   id: 1,                                                     │
   │   prompt_len: 512,      # input tokens                       │
   │   output_len: 128,      # expected output tokens             │
   │   arrival_time: 0.0     # timestamp                          │
   │ }                                                            │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. Scheduler Decision
   ┌─────────────────────────────────────────────────────────────┐
   │ Scheduler.get_next_batch(                                    │
   │   max_batch_size=32,                                         │
   │   available_memory=16GB                                      │
   │ )                                                            │
   │                                                              │
   │ Returns: [Request_1, Request_5, Request_8, ...]              │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. Memory Allocation
   ┌─────────────────────────────────────────────────────────────┐
   │ KVCache.allocate(request)                                    │
   │                                                              │
   │ memory_needed = prompt_len × bytes_per_token                 │
   │               = 512 × 32 × 2 × 4096 × 2                      │
   │               = 256 MB (for 32-layer, 4096-dim model)        │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. Token Generation Loop
   ┌─────────────────────────────────────────────────────────────┐
   │ for step in range(output_len):                               │
   │     # Generate one token for each request in batch           │
   │     tokens = engine.step(batch)                              │
   │                                                              │
   │     # Extend KV cache                                        │
   │     for request in batch:                                    │
   │         kv_cache.extend(request, 1)                          │
   │                                                              │
   │     # Check for completion                                   │
   │     for request in batch:                                    │
   │         if request.is_complete():                            │
   │             metrics.record(request)                          │
   │             kv_cache.free(request)                           │
   └─────────────────────────────────────────────────────────────┘
Scheduling Algorithms
1. FCFS (First-Come-First-Served)
File: memsched/schedulers/fcfs.py


Queue: [R1] → [R2] → [R3] → [R4]
        ↑
      Process in arrival order
The simplest scheduling strategy that processes requests in arrival order.

Aspect	Description
Strategy	Process requests in arrival order
Pros	Fair, no starvation, simple implementation
Cons	Head-of-line blocking, long requests delay short ones
Best for	Low-load scenarios with similar request sizes
2. SJF (Shortest Job First)
File: memsched/schedulers/sjf.py


Queue: [R3:50] → [R1:100] → [R4:200] → [R2:500]
        ↑
      Sorted by estimated output length
Prioritizes requests with shorter estimated output lengths.

Aspect	Description
Strategy	Sort by estimated output length, process shortest first
Pros	Minimizes average waiting time
Cons	Requires output length prediction, may starve long requests
Best for	Workloads with high variance in request sizes
3. MLFQ (Multi-Level Feedback Queue)
File: memsched/schedulers/mlfq.py


Priority 0 (Highest): [R1] [R5]     ← New requests start here
Priority 1:           [R2] [R3]     ← Demoted after using time quantum
Priority 2:           [R4]          ← Further demoted
Priority 3 (Lowest):  [R6] [R7]     ← Long-running requests
Uses multiple priority queues with feedback-based priority adjustment.

Aspect	Description
Strategy	Multiple queues with priority demotion
Pros	Balances responsiveness and throughput
Cons	More complex, requires tuning
Parameters	num_queues=4, time_quantum=10
4. vLLM-Style Scheduler
File: memsched/schedulers/vllm_scheduler.py


┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Waiting   │───→│   Running   │───→│  Completed  │
│    Queue    │    │    Batch    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       ↑                  │
       │                  │ Memory pressure
       │                  ▼
       │           ┌─────────────┐
       └───────────│  Preempted  │
                   │   (Swap)    │
                   └─────────────┘
Implements continuous batching with memory-aware preemption, inspired by vLLM.

Aspect	Description
Strategy	Continuous batching + preemption under memory pressure
Pros	High throughput, efficient memory utilization
Cons	Preemption overhead when memory is tight
Key Feature	Swap out KV cache when memory exhausted
Reference: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023

5. MemSched (Memory-Aware Scheduler) - Ours
File: memsched/schedulers/memsched_scheduler.py


┌─────────────────────────────────────────────────────────────┐
│                    MemSched Decision Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Check Memory State                                      │
│     ┌─────────────────────────────────────────────────┐    │
│     │ memory_ratio = used_memory / total_memory        │    │
│     │ if memory_ratio > 0.9: HIGH_PRESSURE             │    │
│     │ if memory_ratio > 0.7: MEDIUM_PRESSURE           │    │
│     │ else: LOW_PRESSURE                               │    │
│     └─────────────────────────────────────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│  2. Compute Request Priority                                │
│     ┌─────────────────────────────────────────────────┐    │
│     │ priority = w1 × completion_ratio                 │    │
│     │          + w2 × (1 / memory_needed)              │    │
│     │          + w3 × wait_time                        │    │
│     └─────────────────────────────────────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│  3. Adaptive Batch Sizing                                   │
│     ┌─────────────────────────────────────────────────┐    │
│     │ if HIGH_PRESSURE: batch_size = min_batch         │    │
│     │ if MEDIUM_PRESSURE: batch_size = base_batch / 2  │    │
│     │ if LOW_PRESSURE: batch_size = max_batch          │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
Aspect	Description
Strategy	Proactive memory management + intelligent prioritization
Pros	Best memory efficiency, minimal preemptions
Cons	Higher scheduling overhead
Key Features	Memory pressure prediction, completion-aware priority
Module Interactions

┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN SIMULATION LOOP                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│    Workload     │         │    Scheduler    │         │   LLM Engine    │
│    Generator    │         │                 │         │                 │
├─────────────────┤         ├─────────────────┤         ├─────────────────┤
│                 │ Request │                 │  Batch  │                 │
│ generate()  ────┼────────→│ add_request()   │         │                 │
│                 │         │                 │         │                 │
│                 │         │ get_next_batch()├────────→│ step()          │
│                 │         │        ↑        │         │    │            │
│                 │         │        │        │         │    ▼            │
│                 │         │        │        │  Status │ ┌────────────┐  │
│                 │         │        └────────┼─────────┤ │  KVCache   │  │
│                 │         │                 │         │ └────────────┘  │
│                 │         │ update_request()│←────────┤                 │
│                 │         │                 │  Done   │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                       │
                                       │ Metrics
                                       ▼
                            ┌─────────────────┐
                            │ MetricsCollector│
                            ├─────────────────┤
                            │ record()        │
                            │ summarize()     │
                            └─────────────────┘
API Between Modules
python

# Scheduler → Engine
batch = scheduler.get_next_batch(
    max_batch_size=32,
    available_memory=engine.kv_cache.available()
)

# Engine → KVCache
for request in batch:
    if request.is_prefill:
        kv_cache.allocate(request.id, request.prompt_len)
    else:
        kv_cache.extend(request.id, tokens=1)

# Engine → Scheduler (feedback)
for request in batch:
    scheduler.update_request(
        request,
        tokens_generated=1,
        is_complete=(request.generated >= request.output_len)
    )

# Scheduler → Engine (preemption)
if memory_pressure:
    victims = scheduler.select_preemption_victims()
    for victim in victims:
        kv_cache.evict(victim.id)
        scheduler.requeue(victim)
Installation
bash

git clone https://github.com/eldricwang/LLMServer-Scheduling.git
cd LLMServer-Scheduling
pip install -e .
Quick Start
bash

# Run comparison of all schedulers
python experiments/run_comparison.py

# Run with specific workload
python experiments/run_comparison.py --workload bursty --num_requests 500
Extending to Real Systems
Architecture Comparison

┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│         SIMULATOR               │    │        REAL SYSTEM              │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│                                 │    │                                 │
│  WorkloadGenerator              │    │  HTTP/gRPC Server               │
│  (synthetic requests)           │    │  (real user requests)           │
│           │                     │    │           │                     │
│           ▼                     │    │           ▼                     │
│  Scheduler ←──────────────────────────→ Scheduler (SAME)               │
│           │                     │    │           │                     │
│           ▼                     │    │           ▼                     │
│  SimulatedEngine                │    │  RealEngine                     │
│  (time.sleep)                   │    │  (model.forward)                │
│           │                     │    │           │                     │
│           ▼                     │    │           ▼                     │
│  SimulatedKVCache               │    │  GPUKVCache                     │
│  (counter)                      │    │  (torch.Tensor on CUDA)         │
│                                 │    │                                 │
└─────────────────────────────────┘    └─────────────────────────────────┘
Step 1: Replace Token Generator
python

# Simulation (current)
class SimulatedEngine:
    def step(self, batch):
        time.sleep(len(batch) * self.time_per_token)
        return [random_token() for _ in batch]

# Real System
class RealEngine:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.cuda()
    
    def step(self, batch):
        input_ids = self.prepare_inputs(batch)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        return outputs.logits.argmax(dim=-1)
Step 2: Replace KV Cache
python

# Simulation (current)
class SimulatedKVCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.used = 0
    
    def allocate(self, request_id, num_tokens):
        size = num_tokens * self.bytes_per_token
        self.used += size

# Real System
class GPUKVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_tokens):
        self.cache = torch.empty(
            (num_layers, 2, max_tokens, num_heads, head_dim),
            dtype=torch.float16,
            device='cuda'
        )
        self.allocated = {}  # request_id -> (start, length)
Step 3: Add API Server
python

from fastapi import FastAPI
from memsched.schedulers import MemSchedScheduler

app = FastAPI()
scheduler = MemSchedScheduler()
engine = RealEngine("meta-llama/Llama-2-7b")

@app.post("/v1/completions")
async def generate(prompt: str, max_tokens: int = 100):
    request = Request(
        prompt=tokenize(prompt),
        max_output_len=max_tokens
    )
    scheduler.add_request(request)
    
    # Async wait for completion
    result = await request.completion_event.wait()
    return {"text": detokenize(result)}
Integration Checklist
Component	Simulation	Real System
Request Source	WorkloadGenerator	HTTP API
Scheduler	✓ Same code	✓ Same code
Token Generation	time.sleep()	model.forward()
KV Cache	Memory counter	GPU tensors
Tokenization	Length only	Real tokenizer
Output	Metrics file	HTTP response
Project Structure

memsched/
├── memsched/
│   ├── core/
│   │   ├── request.py          # Request dataclass
│   │   ├── kv_cache.py         # KV cache simulation
│   │   └── engine.py           # LLM engine simulation
│   ├── schedulers/
│   │   ├── base.py             # Abstract base class
│   │   ├── fcfs.py             # First-Come-First-Served
│   │   ├── sjf.py              # Shortest Job First
│   │   ├── mlfq.py             # Multi-Level Feedback Queue
│   │   ├── vllm_scheduler.py   # vLLM-style continuous batching
│   │   └── memsched_scheduler.py  # Our memory-aware scheduler
│   └── simulator/
│       ├── simulator.py        # Main simulation loop
│       ├── metrics.py          # Metrics collection
│       └── workload.py         # Workload generation
├── experiments/
│   └── run_comparison.py       # Benchmark script
├── setup.py
└── README.md
License
MIT License

References
vLLM - Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. GitHub

Orca - Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022.

FastServe - Wu et al., "Fast Distributed Inference Serving for Large Language Models", 2023.

MLFQ - Arpaci-Dusseau, "Operating Systems: Three Easy Pieces", Chapter 8.