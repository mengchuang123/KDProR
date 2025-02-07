import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        query = self.split_heads(self.wq(query), batch_size)  
        key = self.split_heads(self.wk(key), batch_size)    
        value = self.split_heads(self.wv(value), batch_size) 
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value)
        
        # Combining heads to original shape
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.dense(scaled_attention), attention_weights

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        d_k = key.size(-1)
        scaled_attention_logits = matmul_qk / d_k**0.5
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    


class PrototypeMatch(nn.Module):
    def __init__(self, prototype_num, vid_dim):
        super(PrototypeMatch, self).__init__()
        self.k = prototype_num
        self.linear = nn.Linear(vid_dim, self.k)  #
        self.relu = nn.ReLU()

    def forward(self, inputs):
        #inputs [batch_size, frame, vid_dim]
        protoweight = self.relu(self.linear(inputs))  # [batch_size, frame, k]
        protoweight = protoweight.transpose(1,2)  # [batch_size, k, frame]
        result = torch.bmm(protoweight, inputs) 
        return protoweight, result  # [batch_size, k, vid_dim]
