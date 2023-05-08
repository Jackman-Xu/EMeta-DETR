import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ops.modules import MSDeformAttn, MSDeformAttn_split
from models.ecam_attention import SingleHeadSiameseAttention
from models.CDETR_attn import Modified_MultiheadAttention
from models.adaptive_fusion import AdaptiveFusion

from util.misc import inverse_sigmoid


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4):
        super().__init__()

        # ❗️❗️❗️找个地方，可以是在main.py中声明，d_model必须要能够整除以num_head
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layers = nn.ModuleList()
        ECAM_position = 0 # ECAM所在的encoder layer层，0～5
        for i in range(num_encoder_layers):
            is_before_ECAM = i < ECAM_position
            encoder_layers.append(
                DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                  dropout, activation,
                                                  num_feature_levels, nhead, enc_n_points, QSAttn=(i == ECAM_position), is_before_ECAM=is_before_ECAM) # 这里是当为Encoder layer第一层时，QSAttn为True，使用作者设计的CAM关系聚合模块。
            )
        self.emeta_encoder = DeformableTransformerEncoder(encoder_layers, num_encoder_layers)


        decoder_layer = DeformableTransformerDecoderLayer(2 * d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)

        self.emeta_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate=return_intermediate_dec)

        if self.num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self): # 重置参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        if self.num_feature_levels > 1:
            normal_(self.level_embed)

    def get_valid_ratio(self, mask): # 确定图像有效区域与总长度的比值，因为图像resize之后会有pad，所以加入mask来补齐
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, class_prototypes, tsp):
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape # src是backbone输入得到的特征图
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # （bs, c, hw）
            mask = mask.flatten(1) # (_, hxw)
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # （bs, c, hw）
            if self.num_feature_levels > 1:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) # 位置编码+尺度编码
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # ********************  Encoder  ********************
        memory = self.emeta_encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, class_prototypes, tsp
        )

        # prepare input for decoder
        bs, _, _ = memory.shape
        query_embed, tgt = torch.split(query_embed, self.d_model, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = torch.cat([tgt, tgt], dim=-1) # 将tgt初始化改造为2倍d_model，以适应decoder的输入
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid() # 将（batchsize, num_object_query, d_model） -> （batchsize, num_object_query, 2)，再进行sigmoid，即将object_queries映射到memory图上的坐标点
        init_reference_out = reference_points

        # ********************  Decoder  ********************
        # Here, decoder is modified to meta-decoder
        # Category-agnostic transformer decoder
        
        hs, inter_references = self.emeta_decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten
        )
        # reference points是query_embed进行线性映射到2D值上，并进行sigmoid的值
        # memory是Encoder部分输出的最后的特征图

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out
    


    # 提取instance prototypes
    def forward_supp_branch(self, srcs, masks, pos_embeds, support_boxes): # support branch最终是返回class_prototypes
        # 一次传入的srcs是episode_size张support图片

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        class_prototypes, feat_before_RoIAlign_GAP = self.encoder.forward_supp_branch(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, support_boxes
        )

        # class_prototypes = （episode_size, d_model）
        return class_prototypes, feat_before_RoIAlign_GAP



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,  # QSAttn指示是否该层是CAM模块所在的Encoder layer层，is_before_ECAM指示现在是否在ECAM所在的Encoder layer层之前（不包括ECAM层）
                 d_model=128, d_ffn=512,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, QSAttn=False, is_before_ECAM=True):
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.QSAttn = QSAttn
        self.is_before_ECAM = is_before_ECAM

        # self attention
        if self.is_before_ECAM or self.QSAttn: # 要么是在ECAM层之前，要么是在ECAM层，执行原始的MSDeformAttn操作
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        else: # 如果是在ECAM交互之后的，那么维度因为拼接操作，扩展到了2 * d_model，并且需要使用自己编写的分通道映射操作的MSDeformAttn_split方法
            self.self_attn = MSDeformAttn_split(2 * d_model, n_levels, n_heads, n_points) # Encoder正常的deformable attention模块
            self.norm1_another = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

       
        # siamese attention
        if self.QSAttn: # 如果QSAttn为True，使用CAM模块
            # # 自适应融合模块
            # self.adaptive_fusion = AdaptiveFusion(d_model)

            # E-CAM模块 
            self.siamese_attn = SingleHeadSiameseAttention(d_model)

        # ffn  
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        if not self.is_before_ECAM: # 如果当前是在ECAM层，或者是在ECAM层之后，新建另外1个FFN分支
            self.linear1_another = nn.Linear(d_model, d_ffn)
            self.linear2_another = nn.Linear(d_ffn, d_model)
            self.norm3_another = nn.LayerNorm(d_model)


    @staticmethod
    def with_pos_embed(tensor, pos): # 融合位置编码信息
        if pos is None:
            return tensor
        elif tensor.shape[-1] == pos.shape[-1]:
            return tensor + pos
        else:
            return tensor + torch.cat([pos, pos], dim=-1)
    

    def forward_ffn(self, src): # 如果在ECAM层之前（不包括ECAM层），直接用原来的代码即可
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        src = self.norm3(src)
        return src


    def forward_ffn_split(self, src): # 如果是在ECAM层或者ECAM层之后的Encoder layer，需要分开使用不同的FFN处理不同的通道
        src1 = src[:, :, :self.d_model]
        src2 = src[:, :, self.d_model:]

        src1 = self.activation(self.linear1(src1))
        src2 = self.activation(self.linear1_another(src2))
        src_cat12 = self.dropout3(torch.cat([src1, src2], dim=-1))

        src_1 = self.linear2(src_cat12[:, :, :self.d_ffn])
        src_2 = self.linear2_another(src_cat12[:, :, self.d_ffn:])
        src = src + self.dropout4(torch.cat([src_1, src_2], dim=-1))

        src1 = self.norm3(src[:, :, :self.d_model])
        src2 = self.norm3_another(src[:, :, self.d_model:])
        
        src = torch.cat([src1, src2], dim=-1)

        return src


    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask, class_prototypes, tsp):
    # def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask, category_codes, tsp):
        # self attention，传入偏移点以及src，src代表不同尺度的特征图拉平汇总
        src_sa = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src_sa) # 注意力机制计算->dropout->残差链接->norm
        
        if self.is_before_ECAM or self.QSAttn:  # 如果是在ECAM层及ECAM层之前，则使用原操作
            src = self.norm1(src)

        else: # 如果是在ECAM层之后，src因为拼接导致是两倍的通道
            src1 = src[:, :, :self.d_model]
            src2 = src[:, :, self.d_model:]
            src1 = self.norm1(src1)
            src2 = self.norm1_another(src2)
            src = torch.cat([src1, src2], dim=-1)


        if self.QSAttn: # 如果有QSAttn的话，那就是先进行正常的MSDeformAttn，然后进行作者设计的CAM模块的注意力计算，再进行FFN层的处理
            # ECAM模块 siamese attention
            src = self.siamese_attn(src, class_prototypes, tsp) # 分别对应q, k, tsp
        

        # ffn
        if self.is_before_ECAM:
            src = self.forward_ffn(src)
        else:
            src = self.forward_ffn_split(src)

        return src


    # support分支的encoder layer的前向传播 (support分支全部执行完整的self-attn+FFN操作，并且每一层都会返回该层的GAP得到的class prototype)
    def forward_supp_branch(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask, support_boxes):
        # self attention，先进行正常的MSDeformAttn
        src_sa = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src_sa)
        src = self.norm1(src) # 得到的特征图信息是不包含位置信息pos的，而是完全与内容相关的信息
        
        feat_before_RoIAlign_GAP = src

        # RoIAlign + GAP操作来获取instance prototypes
        support_img_h, support_img_w = spatial_shapes[0, 0], spatial_shapes[0, 1]
        class_prototypes = torchvision.ops.roi_align(
            src.transpose(1, 2).reshape(src.shape[0], -1, support_img_h, support_img_w), # N, C, support_img_h, support_img_w
            support_boxes,
            output_size=(7, 7),
            spatial_scale=1 / 32.0, # 取的support特征图是1/32的尺度，也就是把support_boxes映射到当前尺度的特征图上
            aligned=True).mean(3).mean(2) # 进行2D平面的GAP，  K, C，其中K表示bbox的个数
        
        # ffn
        src = self.forward_ffn(src)
        
        return src, class_prototypes, feat_before_RoIAlign_GAP




class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layers, num_layers):
        super().__init__()
        self.layers = encoder_layers
        self.num_layers = num_layers
        assert self.num_layers == len(self.layers)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): # 获取reference points
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # 过滤掉那些本身就被mask掉的reference points
        return reference_points # 返回为2D特征图构造的坐标图，之后会和query所预测的offset偏移量进行坐标偏移，进而得到sampling_localtions

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask, class_prototypes, tsp):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        for i, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, class_prototypes, tsp)
        return output

    def forward_supp_branch(self, src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask, support_boxes):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        for i, layer in enumerate(self.layers):
            if layer.is_before_ECAM or layer.QSAttn: # 执行到ECAM处即可
                output, class_prototypes, feat_before_RoIAlign_GAP = layer.forward_supp_branch(
                    output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, support_boxes
                )
        return class_prototypes, feat_before_RoIAlign_GAP # class_prototypes为(episode_size, d_model)， feat_before_RoIAlign_GAP (episode_size, hw, d_model)



class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=512,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        # 这里的d_model是前后拼接的的两个向量（128d）的总的通道长度
        # d_ffn的大小应该是half_d_model的4倍，也就是128*4=512
        super().__init__()

        self.d_model = d_model
        self.half_d_model = d_model // 2
        self.n_heads = n_heads
        self.d_ffn = d_ffn

        # self attention
        self.linear_q1 = nn.Linear(self.half_d_model, self.half_d_model)
        self.linear_k1 = nn.Linear(self.half_d_model, self.half_d_model)
        self.linear_v1 = nn.Linear(self.half_d_model, self.half_d_model)

        self.linear_q2 = nn.Linear(self.half_d_model, self.half_d_model)
        self.linear_k2 = nn.Linear(self.half_d_model, self.half_d_model)
        self.linear_v2 = nn.Linear(self.half_d_model, self.half_d_model)

        self.self_attn = Modified_MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1_1 = nn.LayerNorm(self.half_d_model)
        self.norm1_2 = nn.LayerNorm(self.half_d_model)


        # ==============================================================================
        # cross attention
        self.cross_attn = MSDeformAttn_split(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2_1 = nn.LayerNorm(self.half_d_model)
        self.norm2_2 = nn.LayerNorm(self.half_d_model)
        # ==============================================================================
        
        

        # ffn
        self.linear1_1 = nn.Linear(self.half_d_model, d_ffn)
        self.linear1_2 = nn.Linear(self.half_d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        
        self.linear2_1 = nn.Linear(d_ffn, self.half_d_model)
        self.linear2_2 = nn.Linear(d_ffn, self.half_d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3_1 = nn.LayerNorm(self.half_d_model)
        self.norm3_2 = nn.LayerNorm(self.half_d_model)


    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        elif tensor.shape[-1] == pos.shape[-1]:
            return tensor + pos
        else:
            return tensor + torch.cat([pos, pos], dim=-1)


    def forward_ffn_split(self, tgt1, tgt2): # tgt1，tgt2各自进行自己的FFN操作，最终拼在一起输出
        tgt_1 = self.activation(self.linear1_1(tgt1))
        tgt_2 = self.activation(self.linear1_2(tgt2))
        tgt_cat12 = self.dropout3(torch.cat([tgt_1, tgt_2], dim=-1))

        tgt_1 = self.linear2_1(tgt_cat12[:, :, :self.d_ffn])
        tgt_2 = self.linear2_2(tgt_cat12[:, :, self.d_ffn:])
        tgt_cat12 = self.dropout4(torch.cat([tgt_1, tgt_2], dim=-1))

        tgt1 = tgt1 + tgt_cat12[:, :, :self.half_d_model]
        tgt2 = tgt2 + tgt_cat12[:, :, self.half_d_model:]

        tgt1 = self.norm3_1(tgt1)
        tgt2 = self.norm3_2(tgt2)

        return torch.cat([tgt1, tgt2], dim=-1)

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask):
        # query_pos的维度需要是tgt的一半
        batchsize, num_queries, _ = tgt.shape

        # self attention
        tgt_add_pos = self.with_pos_embed(tgt, query_pos) # 位置编码

        q1 = k1 = tgt_add_pos[:, :, :self.half_d_model]
        q2 = k2 = tgt_add_pos[:, :, self.half_d_model:]

        tgt1, tgt2 = tgt[:, :, :self.half_d_model], tgt[:, :, self.half_d_model:] # 对半拆分通道
        
        # 两半各自通道完成各自的线性映射，并按照每个head内不同通道堆叠
        # 这样可以得到例如(1,2),(1,2),(1,2),(1,2)，这样能够保证同个head计算出来的attn的性质是等价的，共同利用了1，2的特征
        # 如果直接用(1111),(2222)，这样去计算的话，前面的head计算的attn就关注前面的特征，而后面的head计算得到的attn就只关注后面的特征
        q1 = self.linear_q1(q1).view(batchsize, num_queries, self.n_heads, self.half_d_model//self.n_heads)
        q2 = self.linear_q2(q2).view(batchsize, num_queries, self.n_heads, self.half_d_model//self.n_heads)
        q = torch.cat([q1, q2], dim=-1).view(batchsize, num_queries, self.d_model)
        
        k1 = self.linear_k1(k1).view(batchsize, num_queries, self.n_heads, self.half_d_model//self.n_heads)
        k2 = self.linear_k2(k2).view(batchsize, num_queries, self.n_heads, self.half_d_model//self.n_heads)
        k = torch.cat([k1, k2], dim=-1).view(batchsize, num_queries, self.d_model)

        v1 = self.linear_v1(tgt1).view(batchsize, num_queries, self.n_heads, self.half_d_model//self.n_heads)
        v2 = self.linear_v2(tgt2).view(batchsize, num_queries, self.n_heads, self.half_d_model//self.n_heads)
        v = torch.cat([v1, v2], dim=-1).view(batchsize, num_queries, self.d_model)

        tgt_sa, attn_weights = self.self_attn(q, k, v)
        tgt = tgt + self.dropout1(tgt_sa)
        tgt1 = self.norm1_1(tgt[:, :, :self.half_d_model])
        tgt2 = self.norm1_2(tgt[:, :, self.half_d_model:])
        
        tgt = torch.cat([tgt1, tgt2], dim=-1).view(batchsize, num_queries, self.d_model)


        # cross attention
        tgt_ca = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout2(tgt_ca)
        tgt1 = self.norm2_1(tgt[:, :, :self.half_d_model])
        tgt2 = self.norm2_2(tgt[:, :, self.half_d_model:])


        # ffn
        tgt = self.forward_ffn_split(tgt1, tgt2) # tgt1，tgt2均为(batchsize, num_queries, self.half_d_model)

        
        return tgt # (batchsize, num_queries, self.d_model)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement
        self.bbox_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos, src_padding_mask):
        output = tgt # 首次传入decoder时，tgt和query_pos是一致的，都是（batchsize, num_object_query, d_model）

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None] # 计算出object query映射到到memory图上的坐标，之后会调用DeformAttn方法，在其中会预测出offset偏移量，通过offset偏移量以及memory上的坐标点，完成最终sampling_points的确定。
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None: # bbox的逐层细化的实现⬇️
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate: # 记录中间层的输出结果  以及  reference_points
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points # 返回decoder的输出结果，以及DeformAttn关注的参考点坐标




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu/glu/leaky_relu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
    )
