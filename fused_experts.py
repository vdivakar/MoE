import torch
import torch.nn as nn
from topk_softmax import fused_topk_softmax
import numpy as np

M = 10
E = 8
topk = 2

hidden_size = 4096
model_intermediate_size = 14336 

BLOCK_SIZE_M = 3


# intermediate_cache1 = torch.empty((M, topk, N),
#                                     device=hidden_states.device,
#                                     dtype=hidden_states.dtype)
# intermediate_cache2 = torch.empty((M * topk, N // 2),
#                                     device=hidden_states.device,
#                                     dtype=hidden_states.dtype)
# intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
#                                     device=hidden_states.device,
#                                     dtype=hidden_states.dtype)

class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = torch.nn.Linear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False)
        self.w2 = torch.nn.Linear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False)
        self.w3 = torch.nn.Linear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

experts = nn.ModuleList([
            MixtralMLP(E,
                       hidden_size,
                       model_intermediate_size,)
            for _ in range(E)
        ])

def vanilla_implementation(hidden_states):
    sequence_length, hidden_dim = hidden_states.shape

    # router_logits: (batch * sequence_length, n_experts)
    gating_output = np.random.random((M, E))

    topk_weights, topk_expert_ids = fused_topk_softmax(hidden_states, gating_output, topk, renormalize=True)


    final_hidden_states = None
    for expert_idx in range(0, E):
        expert_layer = experts[expert_idx]
        expert_mask = (topk_expert_ids == expert_idx)
        expert_weights = torch.tensor((topk_weights * expert_mask).sum(axis=-1, keepdims=True))

        # current_hidden_states = np.matmul(hidden_states, expert_layer)
        current_hidden_states = expert_layer(hidden_states).mul_(expert_weights)
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states.add_(current_hidden_states)

    # return tensor_model_parallel_all_reduce(final_hidden_states).view(
    #     batch_size, sequence_length, hidden_dim)

if __name__=="__main__":
    np.random.seed(0)
    M = 20
    hidden_size = 4096
    E = 8
    topk = 2
    # hidden_states = np.random.rand(M, hidden_size)
    hidden_states = torch.rand(M, hidden_size)
    vanilla_implementation(hidden_states)