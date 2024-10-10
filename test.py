from math import ceil
import numpy as np

x =  [
        [1, 3],
        [2, 5],
        [3, 1],
        [6, 8],
        [3, 4],
        [3, 3],
        [3, 3],
        [3, 3]    
    ]
topk_ids = np.array(x)
BM = 1
E = np.max(topk_ids) + 1


topk_idx_sorted = np.argsort(topk_ids.flatten())
print(f"BM: {BM} | E: {E}")
print(topk_idx_sorted)

# sorted_ids = np.ones((topk_ids.size + (BM-1)*E), dtype=int) * topk_ids.size
max_possible_tokens_post_padding = topk_ids.size + max(1,(BM-1))*E
max_rows = ceil(max_possible_tokens_post_padding / BM)
print(f"max_possible_post_pad: {max_possible_tokens_post_padding}")
print(f"max_rows: {max_rows}")
sorted_ids_mat = np.ones((max_rows, BM), dtype=int) * topk_ids.size
expert_idx_map = np.ones((max_rows), dtype=int) * -1
print(sorted_ids_mat.shape)
'''
[1, 3]
[2]
[1, 3, 5]
[5]
[2]
[4]
[]
[4]
'''
expert_idx = 0
row_idx = 0
col_idx = 0
expert_idx_map[0] = 0
for i, val in enumerate(topk_idx_sorted):
    def is_correct_expert(exp_idx, v):
       return exp_idx==topk_ids.flatten()[v]
    while not is_correct_expert(expert_idx, val):
        row_idx+=1
        col_idx=0
        expert_idx+=1
        if expert_idx < E:
            expert_idx_map[expert_idx] = row_idx
          
    if(col_idx>=BM):
        row_idx+=1
        col_idx=0
    sorted_ids_mat[row_idx][col_idx] = val
    col_idx+=1
        

print(sorted_ids_mat)
print(expert_idx_map)