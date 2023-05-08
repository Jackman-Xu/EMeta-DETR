import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        logits = torch.bmm(q, k.transpose(1, 2))
        logits = logits / self.temperature # logits = qk/temperature, logits = (batchsize, len_q, len_k)

        attn = self.softmax(logits)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn, logits # 返回的是最后注意力计算完的值，dropout之后的attn，dropout前的log_attn值


class SingleHeadSiameseAttention(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""
    def __init__(self, d_model): # Q和K共享映射参数矩阵，V无映射权重矩阵。
        super().__init__()
        self.n_head = 1 # head数量为1，表示单头的注意力机制

        self.d_model = d_model
        self.d_head = d_model // self.n_head
        assert self.n_head * self.d_head == self.d_model

        self.w_qk = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        # nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_model))) # 初始化qk共享的权重矩阵（原Meta-DETR的）
        nn.init.xavier_uniform_(self.w_qk.weight) # 初始化qk共享的权重矩阵，改成xavier_uniform_吧，一般线性分类器的weight都用这个。❗️❗️❗️


        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_head, 0.5)) # Q·K/ √d
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, tsp): # query, class_prototype, task_encodings
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_tsp, _ = tsp.size()

        assert len_k == len_tsp
        residual = q # b x len_q x d_model

        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_head)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_head)
        tsp = tsp.view(sz_b, len_tsp, self.n_head, self.d_head)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_head)  # (n_head * b) x len_q x d_head
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_head)  # (n_head * b) x len_k x d_head
        tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp, self.d_head)  # (n_head * b) x len_tsp x d_head

        # aggregate the task_encodings with the attention weights calculated by the query feature and class prototypes.
        output, attn, logits = self.attention(q, k, tsp)
        output.view(self.n_head, sz_b, len_q, self.d_head)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x len_q x (n_head * d_head)
        output = self.linear(output) # b x len_q x d_model

        # concat
        output = torch.cat((output, residual), dim=-1) # b x len_q x (2 * d_model)

        # Foreground enhancement
        logits = logits.view(self.n_head, sz_b, len_q, len_k)
        logits = logits.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x len_q x (n_head * len_k)
        logits = logits.sum(dim=-1, keepdim=True) # # b x len_q x 1
        importance = logits.sigmoid() # importance表示前景概率
        

        # enhanced_rate = 2 # 放大前景的效果
        # 或者是用importance内部的值除以一个阈值，如0.75，这样超过0.75的importance值就会得到大于1的值，用该值去增强前景；
        # 而低于0.75的值除以0.75会得到小于1的值，用这个值去削弱背景。
        # importance = importance * enhanced_rate

        output = importance * output # element-wise multiplication

        return output # b x len_q x (2 * d_model)


