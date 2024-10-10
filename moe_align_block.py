from math import ceil
import numpy as np

def align(topk_ids, BM, E):
    '''
    [
        [1, 3],
        [2, 5],
        [3, 1],
        [6, 8],
        [3, 4],    
    ]
    '''
    BS, topk = topk_ids.shape
    list_of_list = []
    sorted_ids = np.zeros((topk_ids.size + (BM-1)*E))
    # num_tokens + (BM-1)*E
    for _ in range(E):
        list_of_list.append([])
    topk_ids_flatten = topk_ids.flatten()
    for i, e in enumerate(topk_ids_flatten):
        list_of_list[e].append(i)

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
    expert_index_list = [None]*E
    global_index = 0
    delimiter = topk_ids.size
    for i in range(E):
        num_tokens = len(list_of_list[i])
        num_expert_rows = ceil(max(1,num_tokens) / BM)
        list_of_list[i] = list_of_list[i] + [delimiter]*(BM*num_expert_rows - num_tokens)
        expert_index_list[i] = global_index
        global_index += num_expert_rows

    for l in list_of_list:
        print(l)
    print(expert_index_list)

    ''' 
    TODO: reshape into BM sized rows
    '''
    # x = np.array(list_of_list)
    # x = x.reshape((BM, -1))
    # print(x)


def align2(topk_ids, BM, E):
    topk_idx_sorted = np.argsort(topk_ids.flatten())
    # print(f"BM: {BM} | E: {E}")
    print(topk_idx_sorted)

    # sorted_ids = np.ones((topk_ids.size + (BM-1)*E), dtype=int) * topk_ids.size
    max_possible_tokens_post_padding = topk_ids.size + max(1,(BM-1))*E
    max_rows = ceil(max_possible_tokens_post_padding / BM)
    # print(f"max_possible_post_pad: {max_possible_tokens_post_padding}")
    # print(f"max_rows: {max_rows}")
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
            # if expert_idx < E:
                # expert_idx_map[expert_idx] = row_idx
            expert_idx_map[row_idx] = expert_idx
            
        if(col_idx>=BM):
            row_idx+=1
            col_idx=0
            if row_idx < max_rows:
                expert_idx_map[row_idx] = expert_idx
        sorted_ids_mat[row_idx][col_idx] = val
        col_idx+=1

    print(sorted_ids_mat)
    print(expert_idx_map)
    return sorted_ids_mat, expert_idx_map

if __name__ == "__main__":

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
    x = np.array(x)
    E = np.max(x) + 1
    BM = 6

    align2(x, BM, E)

'''
Change the approach to using pre-defined
np array. Use sorting and padding the flattened
array as mentioned in the code.
'''