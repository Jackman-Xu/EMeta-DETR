import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deforamble_transformer
from .position_encoding import TaskPositionalEncoding, QueryEncoding


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


 # 本质上就是余弦分类器，比如说，input（3，4），分类器的参数矩阵为（4，8）：
 # input首先会对自身的那3个向量各自进行归一化，变成单位向量（1，4）的形式；
 # 随后分类器的权重会对自身的那8个向量进行各自的归一化，也得到单位向量（1，4）的形式，之后在直接相乘起来，得到的就是余弦相似度。
class distLinear(nn.Module): # 应该是WeightNorm的方法的余弦分类器，加速收敛
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False) # 无bias的线性映射
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) # 为无bias的全连接层设置权重归一化
        self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001) # 对输入x进行规范化处理
        if not self.class_wise_learnable_norm: # 如果没有设置class_wise_learnable_norm，则会手动执行分类器权重在类别维度上的规范化处理
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) # 直接使用无bias的全连接层处理，从indim -> outdim
        scores = self.scale_factor * cos_dist # 结果再乘上scale_factor因子
        return scores


class EMetaDETR(nn.Module):
    """ This is the Meta-DETR module that performs object detection """
    def __init__(self, args, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False): # aux_loss是指使用每层的decoder作为辅助loss
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: the deform-transformer architecture. See deformable_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie, detection slot.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement.
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = args.hidden_dim
        self.num_feature_levels = num_feature_levels

        self.transformer = transformer
        # TaskPositionalEncoding不使用dropout，maxlen设置为episode_size，并且最后调用时，forward函数执行的残差连接为全0的tensor
        self.task_positional_encoding = TaskPositionalEncoding(self.hidden_dim, dropout=0., max_len=self.args.episode_size)
        self.class_embed = nn.Linear(self.hidden_dim, self.args.episode_size) # 最后输出分类时，使用hidden_dim -> episode_size维度的线性映射，完成分类
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3) # 最后输出到坐标值bbox信息时，输入为hidden_dim，隐藏层为hidden_dim，最后一层输出维度为4，一共有3层。
        
        
        if args.class_prototypes_cls_loss: # 当指定class_prototypes_cls_loss时，将hidden_dim映射到num_classes维度上
            if num_feature_levels == 1:
                self.class_prototypes_cls = distLinear(self.hidden_dim, self.num_classes)
            elif num_feature_levels > 1:
                class_prototypes_cls_list = []
                for _ in range(self.num_feature_levels): # 如果指定多尺度的话，对每个尺度都添加一个余弦分类器。
                    class_prototypes_cls_list.append(distLinear(self.hidden_dim, self.num_classes))
                self.class_prototypes_cls = nn.ModuleList(class_prototypes_cls_list)
            else:
                raise RuntimeError

        # self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 4) # 这里可以考虑将query_embed换成原始DeformAttn版本中的embedding可学习的。


        queryencoding = QueryEncoding(self.hidden_dim, dropout=0., max_len=self.num_queries) # decoder部分的object queries的位置编码
        qe = queryencoding() # 目前是不可训练的，固定的参数
        self.query_embed = torch.cat([qe, qe], dim=1) # (max_len, 2d_model)，在dim=1维度下，直接把下一个堆叠到上一个的结尾。
        # 在encoder的query分支的训练时，会把其拆分成query_embed以及tgt，然后各自expand成(batchsize, max_len, 2*d_model)


        if self.num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs): # 这个是适用于backbone如果是输出3个尺度，而想deformable attention那的多尺度是4，就需要再对backbone进行额外的下采样
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1), # 对backbone输出的特征直接通过conv1x1映射到hidden_dim的维度
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.backbone = backbone
        self.with_box_refine = with_box_refine
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0) # 初始化weight和bias
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        

        num_pred = self.transformer.emeta_decoder.num_layers
        if with_box_refine: # 如果是逐层bbox细化的话，则需要将分类器部分进行深复制，并安插在decoder的每层，每层的分类器参数都不同
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.emeta_decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0) # 如果不是逐层bbox细化的话，则进行软复制，每层decoder的分类器共享
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.emeta_decoder.bbox_embed = None

    def forward(self, samples, targets=None, supp_samples=None, supp_class_ids=None, supp_targets=None, class_prototypes=None):
        # train时，samples为4张，supp_samples，25张图片，（episode_num x episode_size）
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        batchsize = samples.tensors.shape[0]
        device = samples.tensors.device

        # During training, class_prototypes are generated from sampled (supp_samples, supp_class_ids, supp_targets)
        if self.training:
            assert supp_samples is not None
            assert supp_class_ids is not None
            assert supp_targets is not None
            # During training stage: we don't have to cover all categories, so there is only 1 episode
            num_support = supp_class_ids.shape[0]
            support_batchsize = self.args.episode_size
            assert num_support == (self.args.episode_size * self.args.episode_num)
            num_episode = self.args.episode_num
            class_prototypes, _ = self.compute_class_prototypes(supp_samples, supp_targets) # 依据25张的supp图和supp标注信息计算class_prototypes，返回的是（num_episode * episode_size, d_model）
        # During inference, class_prototypes should be provided and ready to use for all activated categories
        else:
            assert class_prototypes is not None
            assert supp_class_ids is not None
            # During inference stage: there are multiple episodes to cover all categories, including both base and novel
            num_support = supp_class_ids.shape[0]
            support_batchsize = self.args.episode_size # 这里的episode_size应当去指类别窗口数
            num_episode = math.ceil(num_support / support_batchsize) # 向上取整，例如3/2 = 2




        features, pos = self.backbone(samples) # 训练时，计算出4张query图像的backbone输出的特征图
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs): # 适用在要求为4个及以上的尺度，但是backbone为3尺度的情况
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype) # 调用位置编码处理
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.to(device)

        # To store predictions for each episode
        meta_outputs_classes = []
        meta_outputs_coords = []
        meta_support_class_ids = []

        for i in range(num_episode): # 迭代num_episode中的其中一次
            if self.num_feature_levels == 1:
                if (support_batchsize * (i + 1)) <= num_support:
                    # 最终expand为（batchsize, support_batch_size, d_model）,support_batch_size就是episode_size
                    # episode_class_prototype为（batchsize, support_batch_size, d_model）
                    if self.training:
                        episode_class_prototypes = class_prototypes[(support_batchsize * i): (support_batchsize * (i + 1)), :].unsqueeze(0).expand(batchsize, -1, -1)
                    else:
                        episode_class_prototypes = class_prototypes[:, (support_batchsize * i): (support_batchsize * (i + 1)), :]
                    episode_class_ids = supp_class_ids[(support_batchsize * i): (support_batchsize * (i + 1))]
                else: # 如果当前迭代的episode是最后1个episode，并且最后1个episode的episode_size不是满的
                    if self.training:
                        episode_class_prototypes = class_prototypes[-support_batchsize:, :].unsqueeze(0).expand(batchsize, -1, -1) # 还剩多少取多少
                    else:
                        episode_class_prototypes = class_prototypes[:, -support_batchsize:, :]
                    episode_class_ids = supp_class_ids[-support_batchsize:]
            elif self.num_feature_levels == 4:
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            


            hs, init_reference, inter_references = \
                self.transformer(srcs, masks, pos, query_embeds, episode_class_prototypes, # 每个episode下的训练：输入4张query的特征，以及自身的mask和位置编码信息，再传入query编码、cc（episode_size份的support图的class_prototypes）、taskEncoding信息
                                 self.task_positional_encoding(torch.zeros(self.args.episode_size, self.hidden_dim, device=device)).unsqueeze(0).expand(batchsize, -1, -1))


            # Final FFN to predict confidence scores and boxes coordinates
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]): # hs.shape[0]指decoder的层数level
                if lvl == 0:
                    reference = init_reference.reshape(batchsize, self.num_queries, 2)
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)


                # 这里要拆开，按通道拆成一半⬇️
                outputs_class = self.class_embed[lvl](hs[lvl][:, :, :self.hidden_dim]) # 映射到具体的类别信息，取前一半通道
                tmp = self.bbox_embed[lvl](hs[lvl][:, :, self.hidden_dim:]) # 映射到具体的bbox信息，取后一半通道
                
                
                
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class.view(batchsize, self.num_queries, self.args.episode_size))
                outputs_coords.append(outputs_coord.view(batchsize, self.num_queries, 4))

            meta_outputs_classes.append(torch.stack(outputs_classes)) # (num_episode, decoder_level, batch_size, num_queries, episode_size)
            meta_outputs_coords.append(torch.stack(outputs_coords)) # (num_episode, decoder_level, batch_size, num_queries, 4)
            meta_support_class_ids.append(episode_class_ids) # 记录support的episode_class_ids的信息 # (num_episode, episode_size)

        # Calculate targets for the constructed meta-tasks
        # meta_targets are computed based on original targets and the sampled support images.
        meta_targets = [] # 这里是根据support提供的类别，来过滤掉query中的target信息
        for b in range(batchsize): # 遍历每一张query图片
            for episode_class_ids in meta_support_class_ids: # episode_class_ids指的是每一次提供一个episode_size内的所有class编号
                meta_target = dict()
                target_indexes = [i for i, x in enumerate(targets[b]['labels'].tolist()) if x in episode_class_ids] # 查看query图中哪些label  是在当前episode大小的support图片类别中的，将其过滤选出来，作为当前episode的query的target label。
                meta_target['boxes'] = targets[b]['boxes'][target_indexes]
                meta_target['labels'] = targets[b]['labels'][target_indexes]
                meta_target['area'] = targets[b]['area'][target_indexes]
                meta_target['iscrowd'] = targets[b]['iscrowd'][target_indexes]
                meta_target['image_id'] = targets[b]['image_id']
                meta_target['size'] = targets[b]['size']
                meta_target['orig_size'] = targets[b]['orig_size']
                meta_targets.append(meta_target) # meta_targets得到的为[batchsize * num_episode]个字典，每个字典为1张query对应激活的target信息

        # Create tensors for final outputs
        # default logits are -inf (default confidence scores are 0.00 after sigmoid)
        final_meta_outputs_classes = torch.ones(hs.shape[0], batchsize, num_episode, self.num_queries, self.num_classes, device=device) * (-999999.99)
        final_meta_outputs_coords = torch.zeros(hs.shape[0], batchsize, num_episode, self.num_queries, 4, device=device)
        # Fill in predicted logits into corresponding positions
        class_ids_already_filled_in = []
        for episode_index, (pred_classes, pred_coords, class_ids) in enumerate(zip(meta_outputs_classes, meta_outputs_coords, meta_support_class_ids)):
            # pred_classes  (level, batch_size, num_queries, episode_size)
            # pred_coords  (level, batch_size, num_queries, 4)
            # class_ids  (episode_size)
            for class_index, class_id in enumerate(class_ids): # 当前episode_num的迭代下，对episode_size大小的support class id进行迭代
                # During inference, we need to ignore the classes that already have predictions
                # During training, the same category might appear over different episodes, so no need to filter
                if self.training or (class_id.item() not in class_ids_already_filled_in): # 如果此时是training；或者当前是inference，并且当前的support class是未出现在class_ids_already_filled_in列表中的
                    class_ids_already_filled_in.append(class_id.item())
                    final_meta_outputs_classes[:, :, episode_index, :, class_id] = pred_classes[:, :, :, class_index] # 此处是将分类预测的episode_size个Logits放到了整个num_classes个值列表中，列表为[0,1,..,90]，长度为91，而类别id正好是1～90，所以正好放到对应位置，而0号默认不用，作为哑变量
                    final_meta_outputs_coords[:, :, episode_index, :, :] = pred_coords[:, :, :, :]
        # Pretend we have a batchsize of (batchsize x num_support), and produce final predictions
        final_meta_outputs_classes = final_meta_outputs_classes.view(hs.shape[0], batchsize * num_episode, self.num_queries, self.num_classes) # （decoder_level, batchsize * num_episode, num_queries, num_classes）
        final_meta_outputs_coords = final_meta_outputs_coords.view(hs.shape[0], batchsize * num_episode, self.num_queries, 4) # （decoder_level, batchsize * num_episode, num_queries, 4)

        out = dict()

        out['pred_logits'] = final_meta_outputs_classes[-1] # (batch_size * num_episode, num_queries, num_classes)
        out['pred_boxes'] = final_meta_outputs_coords[-1] # （batch_size * num_episode, num_queries, 4）
        out['activated_class_ids'] = torch.stack(meta_support_class_ids).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1) # （batch_size * num_episode, episode_size）
        out['meta_targets'] = meta_targets  # Add meta_targets into outputs for optimization   （batchsize * num_episode, 1）个字典

        out['batchsize'] = batchsize
        out['num_episode'] = num_episode
        out['num_queries'] = self.num_queries
        out['num_classes'] = self.num_classes

        if self.args.class_prototypes_cls_loss:
            if self.num_feature_levels == 1:
                # out['class_prototypes_cls_logits'] = self.class_prototypes_cls(class_prototypes)
                # out['class_prototypes_cls_targets'] = supp_class_ids
                # TODO: class_prototypes_cls_loss @ every encoder layer! THIS IS ONLY TRIAL!
                #out['class_prototypes_cls_logits'] = self.class_prototypes_cls(torch.cat(class_prototypes, dim=0))
                #out['class_prototypes_cls_targets'] = supp_class_ids.repeat(self.args.dec_layers)

                out['class_prototypes_cls_logits'] = self.class_prototypes_cls(class_prototypes) # 对encoder的第一层layer输出的support特征图的class_prototypes进行distLinear操作，（d_model, num_class）
                # ⬆️ 为(num_episode * episode_size, pred_class)
                out['class_prototypes_cls_targets'] = supp_class_ids # (num_episode * episode_size)
            elif self.num_feature_levels == 4:
                raise NotImplementedError
            else:
                raise NotImplementedError

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(final_meta_outputs_classes, final_meta_outputs_coords)
            for aux_output in out['aux_outputs']:
                aux_output['activated_class_ids'] = torch.stack(meta_support_class_ids).unsqueeze(0).expand(batchsize, -1, -1).reshape(batchsize * num_episode, -1) # （batch_size * num_episode, episode_size）
        return out

    def compute_class_prototypes(self, supp_samples, supp_targets): # 计算support图像的class_prototypes，只在training阶段调用
        num_supp = supp_samples.tensors.shape[0] # 训练时，传入25张support图

        if self.num_feature_levels == 1: # 目前只实现了单尺度的support图像的class_prototypes的计算
            features, pos = self.backbone.forward_supp_branch(supp_samples, return_interm_layers=False) # support图像的backbone特征提取，返回最后1层输出的特征图，如果指定return_interm_layers=True的话，则会返回3个尺度的特征
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src)) # 单尺度情况下，把backbone处理自带的特征映射到规定的hidden_dim下
                masks.append(mask)
                assert mask is not None

            boxes = [box_ops.box_cxcywh_to_xyxy(t['boxes']) for t in supp_targets]
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_sizes = torch.stack([t["size"] for t in supp_targets], dim=0)
            img_h, img_w = img_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            for b in range(num_supp):
                boxes[b] *= scale_fct[b] # bbox的格式化转换，得到实际位置的xyxy的bbox信息


            class_prototypes_list = list()
            feat_before_RoIAlign_GAP_list = list()

            num_episode = math.ceil(num_supp / self.args.episode_size)
            for i in range(num_episode): # 这里我改了，不用num_supp // episode_size，而是用math.ceil
                class_prototypes, feat_before_RoIAlign_GAP = self.transformer.forward_supp_branch([srcs[0][i*self.args.episode_size: (i+1)*self.args.episode_size]], # srcs取0元素是因为现在实现的是单尺度，如果是多尺度，需要思考如何对多尺度特征一起输入DeformDETR中处理
                                                         [masks[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                                                         [pos[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                                                         boxes[i*self.args.episode_size: (i+1)*self.args.episode_size])
                class_prototypes_list.append(class_prototypes) # 这里的transformer.forward_supp_branch传入的support特征图，是以一批一批的episode_size来划分，按顺序传入的
                feat_before_RoIAlign_GAP_list.append(feat_before_RoIAlign_GAP)

                # 也就是最终class_prototypes_list的列表长度为num_episode，列表中每个元素为（episode_size, d_model）

            # 其中每个num_episode下，episode_size内的每个元素都是一个类别的一张图片
            class_prototypes = torch.cat(class_prototypes_list, dim=0)
            feat_before_RoIAlign_GAP = torch.cat(feat_before_RoIAlign_GAP_list, dim=0)
            return class_prototypes, feat_before_RoIAlign_GAP # class_prototypes保存的即是（num_episode * episode_size, d_model），feat_before_RoIAlign_GAP (episode_size, hw, d_model)

        elif self.num_feature_levels == 4:
            raise NotImplementedError
        else:
            raise NotImplementedError
        




    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        # outputs_class = (decoder_layer, batch_size * num_episode, num_queries, num_classes)
        # outputs_coord = (decoder_layer, batch_size * num_episode, num_queries, 4)
        # return => [{}, {}, {}]，列表长度为decoder_layer，列表中每个元素为字典，记载着每层的pred分类和定位结果
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """ This class computes the loss for Meta-DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.matcher = matcher # 传入的匹配器
        self.weight_dict = weight_dict # 存放各个组件的loss的权重
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True): # indices是(batchsize, 2)，其中每一个元素是一个列表，第一个为pred_id（也就是第几个object query），第二个是相应分配的target_class_id
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # (batch_size * num_episode, num_queries, num_classes)
        idx = self._get_src_permutation_idx(indices) # idx为[batch_idx, src_idx]，batch_idx指[0,0,0,0,1,1,1,1,2,2,2,2]，src_idx例如[3,6,8,19, ..., ..., ...]，表示pred_id
        # target：（batchsize * num_episode, 1）个字典
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device) 
        target_classes[idx] = target_classes_o # 依据确定的batch_idx和确定的src_idx，在准确的object query对应的输出位置，将正确的GTlabel分配到该位置上，而其他的位置则默认预测为91，即背景类。（反应的是正负标签分配过程）

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], # 以COCO为例，这里是(batch_size * num_episode, num_queries, 92)
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) # 对应位置标1

        target_classes_onehot = target_classes_onehot[:, :, :-1] # 去掉最后第92的维度，因为上一个scatter操作会把所有背景id为91的值放到92维度并置为1，所以需要去掉最后一个维度，这样剩余的值中，1表示前景，0表示预测为背景。
        # ⬆️ 该操作之后得到真实值的one-hot编码，这几步的代码和Deformable-DETR的保持一致。⬇️ 下面才是作者额外修改的

        # ################### Only Produce Loss for Activated Categories ###################
        activated_class_ids = outputs['activated_class_ids']   # (bs, num_support)
        activated_class_ids = activated_class_ids.unsqueeze(1).repeat(1, target_classes_onehot.shape[1], 1) # (bs, num_queries, num_support)
        loss_ce = sigmoid_focal_loss(src_logits.gather(2, activated_class_ids), # 选择模型处理图片后输出的logits，仅从91个预测值中提取对应episode中的support label对应的项
                                     target_classes_onehot.gather(2, activated_class_ids),# 选择与之对应的target_onehot编码，即最后一个维度是episode大小的0 or 1的值。
                                     num_boxes,
                                     alpha=self.focal_alpha,
                                     gamma=2)

        loss_ce = loss_ce * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # 100 - precision

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none') # L1 loss

        losses = dict()
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag( # giou loss
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_class_prototypes_cls(self, outputs, targets, indices, num_boxes):
        logits = outputs['class_prototypes_cls_logits']
        targets = outputs['class_prototypes_cls_targets']
        losses = {
            "loss_class_prototypes_cls": F.cross_entropy(logits, targets)
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) # i是第i个batch，src是selected prediction id_list，比如说[3, 10, 50, 33]，长度为min(num_queries, object_num)
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = { # 已定义好的不同loss计算方法
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'class_prototypes_cls': self.loss_class_prototypes_cls,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
        """
        # Since we are doing meta-learning over our constructed meta-tasks, the targets for these meta-tasks are
        # stored in outputs['meta_targets']. We dont use original targets.
        targets = outputs['meta_targets'] # 真值
        # out['pred_logits'] (batch_size * num_episode, num_queries, num_classes)
        # out['pred_boxes'] （batch_size * num_episode, num_queries, 4）
        # out['activated_class_ids'] （batch_size * num_episode, episode_size）
        # out['meta_targets'] （batchsize * num_episode, 1）个字典

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) # 匈牙利算法的一对一匹配，得到的indices为（batchsize, 2），其中“2”指(index_i, index_j)，前者对应prediction的id，后者对应target的id

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'class_prototypes_cls':
                        # meta-attention cls loss not for aux_outputs
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        batchsize = outputs['batchsize']
        num_episode = outputs['num_episode']
        num_queries = outputs['num_queries']
        num_classes = outputs['num_classes']

        out_logits = out_logits.view(batchsize, num_episode * num_queries, num_classes)
        out_bbox = out_bbox.view(batchsize, num_episode * num_queries, 4)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1) # 都是(32, 100)，其在每个图片，的num_episode*num_queries*num_classes中，找到概率最大的前100个
        # 并且因为加了topk机制，如何输出也去预测背景类的话，那么top-100取得应该基本全是背景的预测信息，而这是不对的。所以有topk机制的后处理地方，就不能让模型输出的logits包含背景项。
        # Deformable-DETR也是如此。而DETR则是有去预测背景类的，但DETR在后处理部分是仅取max的位置，而不是用topk机制。

        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers): # 多层感知机中，如果不是最后一层layer，都需要加relu激活函数。最后一层线性映射完即可，与FFN类似。
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    if args.dataset_file in ['coco']:
        num_classes = 91
    elif args.dataset_file in ['voc1', 'voc2', 'voc3']:
        num_classes = 21
    else:
        raise ValueError('Unknown args.dataset_file!')

    device = torch.device(args.device)

    backbone = build_backbone(args) # 构建backbone，并结合上位置编码。
    transformer = build_deforamble_transformer(args) # 
    model = EMetaDETR(
        args,
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
    )

    matcher = build_matcher(args) # 初始化时传入args中定义好的cost各自的权重即可，如set_cost_class，set_cost_bbox。实际调用的时候只需要传入pred和target即可

    weight_dict = dict()
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.class_prototypes_cls_loss:
        weight_dict["loss_class_prototypes_cls"] = args.class_prototypes_cls_loss_coef

    losses = ['labels', 'boxes', 'cardinality']

    if args.class_prototypes_cls_loss:
        losses += ["class_prototypes_cls"]

    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

