import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import softmax
import plotly.express as px
import torch
from moe_align_block import align2
from topk_softmax import fused_topk_softmax

def print_arr(arr):
    if torch.is_tensor(arr):
        arr = arr.numpy()
    placeholder = st.empty()
    placeholder.dataframe(arr)

def print_arr_img(arr):
    if torch.is_tensor(arr):
        arr = arr.numpy()
    fig = px.imshow(arr, aspect='equal') #labels
    st.plotly_chart(fig)

M = 10
E = 8
topk = 2

hidden_size = 64 #4096
model_intermediate_size = 128 #14336 

BLOCK_SIZE_M = 5



hidden_states = np.random.rand(M, hidden_size)
gating_output = np.random.random((M, E))

topk_weights, topk_ids = fused_topk_softmax(hidden_states, gating_output, topk, renormalize=True)

st.write("topk_ids")
print_arr(topk_ids)

# moe_align_block_size
sorted_mat, experts_map = align2(topk_ids, BLOCK_SIZE_M, E) 

st.write("sorted_token_ids")
print_arr(sorted_mat)
st.write("expert indices")
print_arr(experts_map.T)

w1 = np.random.rand(E, hidden_size, model_intermediate_size)
w2 = np.random.rand(E, model_intermediate_size, hidden_size)

'''
fused_experts(hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids)
'''

# intermediate_cache1 = torch.empty((M, topk_ids.shape[1], model_intermediate_size),
#                                     device=hidden_states.device, dtype=hidden_states.dtype)
# intermediate_cache2 = torch.empty((M * topk_ids.shape[1], model_intermediate_size // 2),
#                                     device=hidden_states.device, dtype=hidden_states.dtype)
# intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
#                                     device=hidden_states.device, dtype=hidden_states.dtype)

# invoke_fused_moe_kernel(hidden_states,
#                         w1,
#                         intermediate_cache1,
#                         topk_weights,
#                         topk_ids,
#                         sorted_token_ids, # sorted_mat
#                         expert_ids,       # experts_map
#                         num_tokens_post_padded, # TODO. not urgent
#                         False,
#                         topk_ids.shape[1])


'''
    loop over sorted_token_ids rows
    pick the expert from the experts map for each row
    multiply token embeddings and expert
'''
new_hidden_states = np.zeros((M, model_intermediate_size))
for i, row in enumerate(sorted_mat):
    token_ids = row//2
    token_ids = token_ids[token_ids<M]
    selected_tokens_embeddings = hidden_states[token_ids]    
    
    if(experts_map[i] < 0):
        break
    expert_weights = w1[experts_map[i]]
    
    # Do matmul with w2, w3 weights. Include Silu act too 
    new_hidden_states[token_ids] += np.matmul(selected_tokens_embeddings, expert_weights)
    

print_arr_img(new_hidden_states)


'''
TODO:
1. get num_tokens_post_padded from moe_align_block_size. Not urgent though
2. figure out mat-mul operation
    - sorted_token_ids

Done:
    - Vanilla moe implementation

'''