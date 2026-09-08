import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class Expert(nn.Module):
    def __init__(self, hidden_size, ffn_hidden):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, hidden_size)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class SimpleMoE(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        ffn_hidden=4096,
        num_experts=8,
        top_k=2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList([
            Expert(hidden_size, ffn_hidden)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [T, H]
        T, H = x.shape

        router_logits = self.router(x)

        router_probs = F.softmax(
            router_logits,
            dim=-1,
        )

        topk_probs, topk_indices = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )

        # normalize top-k probabilities
        topk_probs = topk_probs / topk_probs.sum(
            dim=-1,
            keepdim=True,
        )

        output = torch.zeros_like(x)

        expert_counts = []

        for expert_id, expert in enumerate(self.experts):

            # Find tokens routed to this expert
            mask = topk_indices == expert_id

            token_idx, k_idx = torch.where(mask)

            expert_counts.append(token_idx.numel())

            if token_idx.numel() == 0:
                continue

            expert_input = x[token_idx]

            expert_output = expert(expert_input)

            weight = topk_probs[
                token_idx,
                k_idx,
            ].unsqueeze(-1)

            output.index_add_(
                0,
                token_idx,
                expert_output * weight,
            )

        return (
            output,
            router_probs,
            topk_indices,
            expert_counts,
        )


def benchmark(model, x, warmup=20, iters=100):

    for _ in range(warmup):
        model(x)

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iters):
        model(x)

    torch.cuda.synchronize()



    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     record_shapes=True,
    #     profile_memory=True,
    # ) as prof:
    
    #     for _ in range(10):
    #         model(x)
    
    # print(
    #     prof.key_averages().table(
    #         sort_by="cuda_time_total",
    #         row_limit=30,
    #     )
    # )


    

    end = time.perf_counter()

    





    

    return (end - start) / iters * 1000


def main():

    device = "cuda"

    T = 4096
    H = 1024
    E = 8
    K = 2
    FFN = 4096

    model = SimpleMoE(
        hidden_size=H,
        ffn_hidden=FFN,
        num_experts=E,
        top_k=K,
    ).to(device)

    x = torch.randn(
        T,
        H,
        device=device,
        dtype=torch.bfloat16,
    )

    model = model.to(torch.bfloat16)

    with torch.no_grad():

        latency = benchmark(
            model,
            x,
        )

        output, probs, indices, counts = model(x)

    counts = torch.tensor(
        counts,
        dtype=torch.float32,
    )

    mean = counts.mean()
    std = counts.std()

    cv = std / mean

    entropy = -(
        probs *
        torch.log(probs + 1e-9)
    ).sum(dim=-1).mean()

    print("=" * 60)

    print(f"Tokens:        {T}")
    print(f"Hidden:        {H}")
    print(f"Experts:       {E}")
    print(f"Top-K:         {K}")
    print(f"FFN hidden:    {FFN}")

    print("-" * 60)

    print(f"Latency:       {latency:.3f} ms")
    print(f"Throughput:    {T / (latency / 1000):.2f} tokens/s")

    print("-" * 60)

    print("Expert load:")

    for i, c in enumerate(counts):
        print(
            f"Expert {i}: {int(c.item()):5d}"
        )

    print("-" * 60)

    print(f"Mean load:     {mean:.2f}")
    print(f"Std load:      {std:.2f}")
    print(f"Load CV:       {cv:.4f}")
    print(f"Router entropy:{entropy.item():.4f}")


if __name__ == "__main__":
    main()