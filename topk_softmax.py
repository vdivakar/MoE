import numpy as np
from scipy.special import softmax
import torch

'''
Input:
    hidden_states : (M, N) # only need to determine 'M', 'device'
    gating_output : (M, E)
    topk          : 2
    renormalize   : True

Output:
    topk_weights : (M, topk)
    topk_ids     : (M, topk)

fused topk softmax (Docs)
 - num experts is a small power of 2
 - assumes k is small, but works for any k

'''

def fused_topk_softmax(hidden_states, gating_output, topk, renormalize):
    M, _ = hidden_states.shape
    # gating_output = np.random.random((M, E))
    gating_output_softmaxed = softmax(gating_output, axis=1)
    topk_weights, topk_ids = torch.topk(torch.from_numpy(gating_output_softmaxed), topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    topk_ids = topk_ids.numpy()
    topk_weights = topk_weights.numpy()
    return topk_weights, topk_ids

if __name__ == "__main__":
    np.random.seed(0)
    M = 20
    hidden_size = 4096
    E = 8
    topk = 2
    hidden_states = np.random.rand(M, hidden_size)
    gating_output = np.random.random((M, E))

    topk_weights, topk_ids = fused_topk_softmax(hidden_states, gating_output, topk, renormalize=True)
    # print(topk_ids)
    # print(topk_ids == 0)
    expert_weights = topk_weights * (topk_ids==0)
    print(torch.tensor(expert_weights).sum(dim=-1, keepdim=True))
    print(topk_ids.shape)