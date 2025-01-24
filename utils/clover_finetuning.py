from torch import svd_lowrank
from copy import deepcopy
import torch
import torch.nn as nn

def orthogonalize_qk_proj(qk_proj, num_heads, head_dim):
    device = qk_proj.weight.device
    dtype = qk_proj.weight.dtype

    qk_weight = deepcopy(qk_proj.weight.data) # (hidden_size, qk_in_dim)
    if qk_proj.bias is not None:
        qk_bias = deepcopy(qk_proj.bias.data).unsqueeze(1) 
        qk_weight = torch.cat([qk_weight, qk_bias],dim=1)  # (hidden_size, qk_in_dim+1)
    
    qk_weight = qk_weight.view(num_heads, head_dim, -1) # (num_heads, head_dim, qk_in_dim) or (num_heads, head_dim, qk_in_dim+1)
    r_weight = []
    for h in range(num_heads):
        Q, R = torch.linalg.qr(qk_weight[h].T.to("cuda").to(torch.float32)) # (qk_in_dim, head_dim), (head_dim, head_dim)
        qk_weight[h] = Q.T.to(device).to(dtype) # (head_dim, v_in_dim) or (head_dim, v_in_dim+1)
        r_weight.append(R.to(device).to(dtype))
        
    r_weight = torch.stack(r_weight)
    if qk_proj.bias is not None:
        qk_proj.bias.data = qk_weight[:,:,-1].reshape(-1).contiguous() # (v_out_dim, )
        qk_weight = qk_weight[:,:,:-1]

    qk_proj.weight.data = qk_weight.reshape(num_heads*head_dim, -1).contiguous()
    
    return r_weight

def orthogonalize_vo_proj(v_proj, o_proj, num_heads, head_dim, niter=16):
    v_device = v_proj.weight.device
    o_device = o_proj.weight.device
    v_dtype = v_proj.weight.dtype
    o_dtype = o_proj.weight.dtype

    v_weight = deepcopy(v_proj.weight.data) # (hidden_size, v_in_dim)
    o_weight = deepcopy(o_proj.weight.data) # (o_out_dim, hidden_size)
    if v_proj.bias is not None:
        v_bias = deepcopy(v_proj.bias.data).unsqueeze(1) # (v_out_dim, 1)
        v_weight = torch.cat([v_weight, v_bias],dim=1)  # (hidden_size, v_in_dim+1)
    
    v_weight = v_weight.view(num_heads, head_dim, -1) # (num_heads, head_dim, v_in_dim) or (num_heads, head_dim, v_in_dim+1)
    o_weight = o_weight.view(-1, num_heads, head_dim) # (o_out_dim, num_heads, head_dim)
    vo_low_rank = torch.einsum('hdv,ohd->hvo', v_weight.to("cuda"), o_weight.to("cuda")).to(torch.float32) # (num_heads, v_in_dim, o_out_dim) or (num_heads, v_in_dim+1, o_out_dim)
    s_weight = []
    for h in range(num_heads):
        if niter <=0:
            U,S,V = torch.linalg.svd(vo_low_rank[h],full_matrices=False)
            U = U[:, :head_dim]
            S = S[:head_dim]
            Vh = V[:head_dim].T
        else:
            U, S, Vh = svd_lowrank(vo_low_rank[h], head_dim, niter=niter)
        v_weight[h] = U.T.to(v_device).to(v_dtype) # (head_dim, v_in_dim) or (head_dim, v_in_dim+1)
        s_weight.append(torch.diag(S).to(v_device).to(v_dtype))
        o_weight[:,h] = Vh.to(o_device).to(o_dtype) # (o_out_dim, head_dim)
        
    s_weight = torch.stack(s_weight)
    if v_proj.bias is not None:
        v_proj.bias.data = v_weight[:,:,-1].reshape(-1).contiguous() # (v_out_dim, )
        v_weight = v_weight[:,:,:-1]

    v_proj.weight.data = v_weight.reshape(num_heads*head_dim, -1).contiguous()
    o_proj.weight.data = o_weight.reshape(-1, num_heads*head_dim).contiguous()
    
    return s_weight

class QKVLinear(nn.Module):
    def __init__(self, linear: nn.Linear, s_weight, num_heads, head_dim, dropout_rate=0.05):
        super().__init__()
        device = linear.weight.device
        dtype = linear.weight.dtype
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.singular_vector = linear
        self.dropout = nn.Dropout(p=dropout_rate)
        self.singular_value = nn.Linear(head_dim , num_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.singular_value.weight.data = s_weight.reshape(num_heads*head_dim, head_dim).contiguous()
    
    def forward(self, x):
        bsz, seq, _ = x.shape
        output = self.singular_vector(x) # (bsz, seq, num_heads*head_dim)
        output = self.dropout(output)
        output = output.view(bsz, seq, self.num_heads, self.head_dim)
        singular_value = self.singular_value.weight.reshape(self.num_heads, self.head_dim, self.head_dim)
        output = torch.einsum("bshd,hde->bshe", output, singular_value)
        output = output.reshape(bsz, seq, self.num_heads*self.head_dim).contiguous()
        return output      
    
def orthogonal_and_replace_qkv_proj(attn: nn.Module, num_heads: int, kv_heads: int, head_dim: int, q_name: str="q_proj", k_name: str="k_proj", v_name: str="v_proj", o_name: str="o_proj", niter=16):
    q_proj = getattr(attn, q_name)
    k_proj = getattr(attn, k_name)
    v_proj = getattr(attn, v_name)
    o_proj = getattr(attn, o_name)
    with torch.no_grad():
        r_weight = orthogonalize_qk_proj(q_proj, num_heads, head_dim)
        setattr(attn, q_name, QKVLinear(q_proj, r_weight, num_heads, head_dim))
        r_weight = orthogonalize_qk_proj(k_proj, kv_heads, head_dim)
        setattr(attn, k_name, QKVLinear(k_proj, r_weight, kv_heads, head_dim))
        if num_heads==kv_heads:
            s_weight = orthogonalize_vo_proj(v_proj, o_proj, kv_heads, head_dim, niter)
        else:
            s_weight = orthogonalize_qk_proj(v_proj, kv_heads, head_dim)
        setattr(attn, v_name, QKVLinear(v_proj, s_weight, kv_heads, head_dim))
        
def merge_qkv_proj(attn: nn.Module, num_heads: int, head_dim: int, qkv_name: str=None):
    with torch.no_grad():
        qkv_proj = getattr(attn, qkv_name)
        assert qkv_proj.singular_vector.bias is None
        singular_vector = qkv_proj.singular_vector.weight.data.reshape(num_heads, head_dim, -1)
        singular_value = qkv_proj.singular_value.weight.data.reshape(num_heads, head_dim, head_dim)
        qkv_weight = torch.einsum("hdD,hde->heD", singular_vector, singular_value)
        qkv_proj.singular_vector.weight.data = qkv_weight.reshape(num_heads*head_dim, -1).contiguous()
        setattr(attn, qkv_name, qkv_proj.singular_vector)