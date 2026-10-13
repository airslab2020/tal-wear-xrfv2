import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm, AffineDropPath
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms
from inertial_baseline.TinyHAR import TinyHAR

class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        model_name,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg               # other cfg for testing
    ):
        super().__init__()
        self.model_name = model_name
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        # generate segment/lable List[N x 2] / List[N] with length = B
        assert video_list[0]['segments'] is not None, "GT action labels does not exist"
        assert video_list[0]['labels'] is not None, "GT action labels does not exist"
        gt_segments = [x['segments'].to(self.device) for x in video_list]
        gt_labels = [x['labels'].to(self.device) for x in video_list]

        # compute the gt labels for cls & reg
        # list of prediction targets
        gt_cls_labels, gt_offsets = self.label_points(points, gt_segments, gt_labels)

        # compute the loss and return
        losses = self.losses(
            fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets
        )

        if self.training:
            return losses
        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return losses, results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results


class TinyHARFeatureExtractor(TinyHAR):
    def __init__(self, *args, **kwargs):
        kwargs["feature_extract"] = True
        super().__init__(*args, **kwargs)

    def forward(self, x):
        _, features = super().forward(x)
        return features


class TinyHAREncoder(nn.Module):
    def __init__(
        self,
        input_dim=30,
        window_size=50,
        project_dim=1024,
        projection_dropout=0.1,
        tinyhar_kwargs=None,
    ):
        super().__init__()
        if tinyhar_kwargs is None:
            tinyhar_kwargs = {}
        filter_num = tinyhar_kwargs.get("filter_num", 64)
        self.window_size = window_size
        self.input_dim = input_dim
        self.raw_feature_dim = 2 * filter_num
        self.project_dim = project_dim
        self.tinyhar = TinyHARFeatureExtractor(
            input_shape=(1, 1, window_size, input_dim),
            number_class=tinyhar_kwargs.get("number_class", 34),
            filter_num=filter_num,
            nb_conv_layers=tinyhar_kwargs.get("nb_conv_layers", 4),
            filter_size=tinyhar_kwargs.get("filter_size", 5),
            cross_channel_interaction_type=tinyhar_kwargs.get(
                "cross_channel_interaction_type", "attn"
            ),
            cross_channel_aggregation_type=tinyhar_kwargs.get(
                "cross_channel_aggregation_type", "FC"
            ),
            temporal_info_interaction_type=tinyhar_kwargs.get(
                "temporal_info_interaction_type", "lstm"
            ),
            temporal_info_aggregation_type=tinyhar_kwargs.get(
                "temporal_info_aggregation_type", "naive"
            ),
            dropout=tinyhar_kwargs.get("dropout", 0.3),
        )
        self.project = nn.Sequential(
            nn.Linear(self.raw_feature_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(projection_dropout),
        )

    def forward(self, imu_windows):
        b, c, t = imu_windows.shape
        assert c == self.window_size * self.input_dim
        imu_seq = imu_windows.permute(0, 2, 1).contiguous()
        imu_seq = imu_seq.view(b * t, self.window_size, self.input_dim)
        feats = self.tinyhar(imu_seq)
        feats = self.project(feats)
        feats = feats.view(b, t, self.project_dim).permute(0, 2, 1).contiguous()
        return feats


class CrossModalIMUFusion(nn.Module):
    def __init__(
        self,
        video_dim=1024,
        imu_dim=1024,
        fusion_dim=512,
        n_head=8,
        dropout=0.1,
        ffn_mult=4,
    ):
        super().__init__()
        self.video_dim = video_dim
        self.imu_dim = imu_dim
        self.fusion_dim = fusion_dim
        self.video_embed = nn.Sequential(
            nn.Conv1d(video_dim, fusion_dim, kernel_size=1),
            LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.imu_embed = nn.Sequential(
            nn.Conv1d(imu_dim, fusion_dim, kernel_size=1),
            LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.cross_attn = nn.MultiheadAttention(
            fusion_dim,
            n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * ffn_mult, fusion_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(fusion_dim)
        self.gate = nn.Conv1d(fusion_dim * 2, fusion_dim, kernel_size=1)

    def forward(self, video_feats, imu_feats, mask):
        assert video_feats.shape[1] == self.video_dim
        assert imu_feats.shape[1] == self.imu_dim
        assert video_feats.shape[-1] == imu_feats.shape[-1]

        video_emb = self.video_embed(video_feats)
        imu_emb = self.imu_embed(imu_feats)
        key_padding_mask = torch.logical_not(mask.squeeze(1))

        video_seq = video_emb.transpose(1, 2)
        imu_seq = imu_emb.transpose(1, 2)
        attn_out, _ = self.cross_attn(
            video_seq,
            imu_seq,
            imu_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused_seq = self.attn_norm(video_seq + self.attn_dropout(attn_out))
        fused_seq = self.ffn_norm(fused_seq + self.ffn(fused_seq))
        fused = fused_seq.transpose(1, 2)

        gate = torch.sigmoid(self.gate(torch.cat([fused, imu_emb], dim=1)))
        fused = gate * fused + (1.0 - gate) * imu_emb
        return fused * mask.to(fused.dtype)


class CrossMHCA(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.0, proj_pdrop=0.0):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.key = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)
        self.proj = nn.Conv1d(n_embd, n_embd, 1)

    def forward(self, x, context, mask):
        b, c, t = x.size()
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)
        q = q.view(b, self.n_head, self.n_channels, t).transpose(2, 3)
        k = k.view(b, self.n_head, self.n_channels, t).transpose(2, 3)
        v = v.view(b, self.n_head, self.n_channels, t).transpose(2, 3)
        att = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = att @ v
        out = out.transpose(2, 3).contiguous().view(b, c, t)
        out = self.proj_drop(self.proj(out))
        if mask is not None:
            out = out * mask.to(out.dtype)
        return out


class CrossTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
    ):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.ln_context = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.attn = CrossMHCA(n_embd, n_head, attn_pdrop, proj_pdrop)
        if n_hidden is None:
            n_hidden = 4 * n_embd
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_embd, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_embd, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, context, mask):
        attn_out = self.attn(self.ln1(x), self.ln_context(context), mask)
        x = x + self.drop_path_attn(attn_out)
        mlp_out = self.mlp(self.ln2(x))
        if mask is not None:
            mlp_out = mlp_out * mask.to(mlp_out.dtype)
        x = x + self.drop_path_mlp(mlp_out)
        return x, mask


class MRAVFFFusion(nn.Module):
    def __init__(self, video_dim, imu_dim, embd_dim, n_head=8, num_levels=6):
        super().__init__()
        self.video_dim = video_dim
        self.imu_dim = imu_dim
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.num_levels = num_levels

        self.video_embed = nn.Sequential(
            nn.Conv1d(video_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
            nn.Conv1d(embd_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )

        self.additional_embed = nn.Sequential(
            nn.Conv1d(imu_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
            nn.Conv1d(embd_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )

        self.feat_merge = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(embd_dim // 2, embd_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(embd_dim // 2, embd_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.feat_merge2 = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(embd_dim // 2, embd_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(embd_dim // 2, embd_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.cross_blocks = nn.ModuleList(
            [
                CrossTransformerBlock(
                    n_embd=embd_dim,
                    n_head=n_head,
                    n_hidden=embd_dim * 4,
                    attn_pdrop=0.1,
                    proj_pdrop=0.1,
                    path_pdrop=0.1,
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, video_feats, imu_feats, mask=None):
        b_v, c_v, t_v = video_feats.shape
        b_i, c_i, t_i = imu_feats.shape
        assert b_v == b_i
        assert t_v == t_i
        assert c_v == self.video_dim
        assert c_i == self.imu_dim

        video_emb = self.video_embed(video_feats)
        imu_emb = self.additional_embed(imu_feats)

        v_levels = []
        a_levels = []
        m_levels = []

        v = video_emb
        a = imu_emb
        m = mask

        for level in range(self.num_levels):
            v_levels.append(v)
            a_levels.append(a)
            m_levels.append(m)
            if level != self.num_levels - 1:
                v = F.max_pool1d(v, kernel_size=2, stride=2, ceil_mode=True)
                a = F.max_pool1d(a, kernel_size=2, stride=2, ceil_mode=True)
                if m is not None:
                    m = F.max_pool1d(
                        m.to(v.dtype), kernel_size=2, stride=2, ceil_mode=True
                    ) > 0.5

        fused_levels = []
        target_len = video_emb.shape[-1]
        for level in range(self.num_levels):
            v_l = v_levels[level]
            a_l = a_levels[level]
            m_l = m_levels[level]
            fused_l, _ = self.cross_blocks[level](v_l, a_l, m_l)
            fused_l = self.feat_merge(fused_l)
            fused_l = v_l + fused_l
            fused_l = self.feat_merge2(fused_l)
            fused_l = v_l + fused_l
            if fused_l.shape[-1] != target_len:
                fused_l = F.interpolate(fused_l, size=target_len, mode="nearest")
            fused_levels.append(fused_l)

        fused = torch.stack(fused_levels, dim=0).mean(dim=0)
        if mask is not None:
            fused = fused * mask.to(fused.dtype)
        return fused


class ConcatProjectFusion(nn.Module):
    def __init__(self, video_dim, imu_dim, embd_dim, dropout=0.1):
        super().__init__()
        self.video_dim = video_dim
        self.imu_dim = imu_dim
        self.proj = nn.Sequential(
            nn.Conv1d(video_dim + imu_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(embd_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )

    def forward(self, video_feats, imu_feats, mask=None):
        assert video_feats.shape[1] == self.video_dim
        assert imu_feats.shape[1] == self.imu_dim
        fused = self.proj(torch.cat([video_feats, imu_feats], dim=1))
        if mask is not None:
            fused = fused * mask.to(fused.dtype)
        return fused


class GateSumFusion(nn.Module):
    def __init__(self, video_dim, imu_dim, embd_dim, dropout=0.1):
        super().__init__()
        self.video_dim = video_dim
        self.imu_dim = imu_dim
        self.video_proj = nn.Sequential(
            nn.Conv1d(video_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )
        self.imu_proj = nn.Sequential(
            nn.Conv1d(imu_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(embd_dim * 2, embd_dim, kernel_size=1),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )

    def forward(self, video_feats, imu_feats, mask=None):
        assert video_feats.shape[1] == self.video_dim
        assert imu_feats.shape[1] == self.imu_dim
        video_emb = self.video_proj(video_feats)
        imu_emb = self.imu_proj(imu_feats)
        gate = self.gate(torch.cat([video_emb, imu_emb], dim=1))
        fused = gate * video_emb + (1.0 - gate) * imu_emb
        if mask is not None:
            fused = fused * mask.to(fused.dtype)
        return fused


class GateConcatFusion(nn.Module):
    def __init__(self, video_dim, imu_dim, embd_dim, dropout=0.1):
        super().__init__()
        self.video_dim = video_dim
        self.imu_dim = imu_dim
        self.video_proj = nn.Sequential(
            nn.Conv1d(video_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )
        self.imu_proj = nn.Sequential(
            nn.Conv1d(imu_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(embd_dim * 2, embd_dim, kernel_size=1),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )
        self.merge = nn.Sequential(
            nn.Conv1d(embd_dim * 2, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(embd_dim, embd_dim, kernel_size=1),
            nn.GroupNorm(16, embd_dim),
            nn.ReLU(),
        )

    def forward(self, video_feats, imu_feats, mask=None):
        assert video_feats.shape[1] == self.video_dim
        assert imu_feats.shape[1] == self.imu_dim
        video_emb = self.video_proj(video_feats)
        imu_emb = self.imu_proj(imu_feats)
        gate = self.gate(torch.cat([video_emb, imu_emb], dim=1))
        fused = self.merge(torch.cat([video_emb, gate * imu_emb], dim=1))
        if mask is not None:
            fused = fused * mask.to(fused.dtype)
        return fused


class SensorAwareGateConcatFusion(GateConcatFusion):
    def __init__(self, video_dim, imu_dim, embd_dim, dropout=0.1, reduction=16):
        super().__init__(video_dim, imu_dim, embd_dim, dropout)
        hidden_dim = max(imu_dim // reduction, 16)
        self.imu_recalibration = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(imu_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, imu_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, video_feats, imu_feats, mask=None):
        imu_feats = imu_feats * self.imu_recalibration(imu_feats)
        return super().forward(video_feats, imu_feats, mask)


class LocalTemporalCrossAttentionGateConcatFusion(nn.Module):
    """Local bidirectional cross-modal attention followed by reliability gating.

    Both modalities are projected to a compact shared space.  At every time
    step, IMU queries a local video window and video queries the corresponding
    local IMU window.  The enhanced streams are gated independently and kept
    separate by concatenation before being passed to the TAL backbone.
    """

    def __init__(
        self,
        video_dim,
        imu_dim,
        embd_dim,
        hidden_dim=256,
        n_heads=4,
        window_size=5,
        dropout=0.1,
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")

        self.video_dim = int(video_dim)
        self.imu_dim = int(imu_dim)
        self.hidden_dim = int(hidden_dim)
        self.window_size = int(window_size)
        self.window_radius = self.window_size // 2

        self.video_proj = nn.Sequential(
            nn.Conv1d(self.video_dim, self.hidden_dim, kernel_size=1),
            nn.GroupNorm(16, self.hidden_dim),
            nn.ReLU(),
        )
        self.imu_proj = nn.Sequential(
            nn.Conv1d(self.imu_dim, self.hidden_dim, kernel_size=1),
            nn.GroupNorm(16, self.hidden_dim),
            nn.ReLU(),
        )

        self.imu_from_video = nn.MultiheadAttention(
            self.hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.video_from_imu = nn.MultiheadAttention(
            self.hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.imu_norm = nn.LayerNorm(self.hidden_dim)
        self.video_norm = nn.LayerNorm(self.hidden_dim)

        self.gate = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 4, self.hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim * 2, self.hidden_dim * 2, kernel_size=1),
            nn.Sigmoid(),
        )
        concat_dim = self.hidden_dim * 2
        self.output_proj = (
            nn.Identity()
            if concat_dim == embd_dim
            else nn.Sequential(
                nn.Conv1d(concat_dim, embd_dim, kernel_size=1),
                nn.GroupNorm(16, embd_dim),
                nn.ReLU(),
            )
        )

    def _local_windows(self, sequence):
        # sequence: B x T x C -> (B*T) x W x C
        sequence = sequence.transpose(1, 2)
        sequence = F.pad(
            sequence, (self.window_radius, self.window_radius), value=0.0
        )
        windows = sequence.unfold(2, self.window_size, 1)
        windows = windows.permute(0, 2, 3, 1).contiguous()
        return windows.view(-1, self.window_size, self.hidden_dim)

    def _local_key_padding_mask(self, valid_mask):
        local_valid = F.pad(
            valid_mask, (self.window_radius, self.window_radius), value=False
        ).unfold(1, self.window_size, 1)
        key_padding_mask = ~local_valid.reshape(-1, self.window_size)

        # MultiheadAttention returns NaNs if every key is masked.  Invalid
        # padded queries are removed by the output mask, so expose their zero
        # windows solely to keep the attention operation numerically finite.
        invalid_queries = ~valid_mask.reshape(-1)
        if invalid_queries.any():
            key_padding_mask[invalid_queries] = False
        return key_padding_mask

    def forward(self, video_feats, imu_feats, mask=None):
        assert video_feats.shape[1] == self.video_dim
        assert imu_feats.shape[1] == self.imu_dim
        assert video_feats.shape[-1] == imu_feats.shape[-1]

        video_emb = self.video_proj(video_feats)
        imu_emb = self.imu_proj(imu_feats)
        video_seq = video_emb.transpose(1, 2)
        imu_seq = imu_emb.transpose(1, 2)
        batch_size, seq_len, _ = video_seq.shape

        if mask is None:
            valid_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=video_feats.device
            )
        else:
            valid_mask = mask.squeeze(1).bool()
        key_padding_mask = self._local_key_padding_mask(valid_mask)
        video_windows = self._local_windows(video_seq)
        imu_windows = self._local_windows(imu_seq)

        imu_context, _ = self.imu_from_video(
            imu_seq.reshape(-1, 1, self.hidden_dim),
            video_windows,
            video_windows,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        video_context, _ = self.video_from_imu(
            video_seq.reshape(-1, 1, self.hidden_dim),
            imu_windows,
            imu_windows,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        imu_enhanced = self.imu_norm(
            imu_seq + imu_context.view(batch_size, seq_len, self.hidden_dim)
        )
        video_enhanced = self.video_norm(
            video_seq + video_context.view(batch_size, seq_len, self.hidden_dim)
        )
        imu_enhanced = imu_enhanced.transpose(1, 2)
        video_enhanced = video_enhanced.transpose(1, 2)

        gates = self.gate(
            torch.cat(
                [imu_enhanced, video_enhanced, imu_emb, video_emb], dim=1
            )
        )
        imu_gate, video_gate = gates.chunk(2, dim=1)
        fused = self.output_proj(
            torch.cat(
                [imu_gate * imu_enhanced, video_gate * video_enhanced], dim=1
            )
        )
        if mask is not None:
            fused = fused * mask.to(fused.dtype)
        return fused


class ResidualMRAVFFFusion(nn.Module):
    def __init__(
        self,
        video_dim,
        imu_dim,
        embd_dim,
        n_head=8,
        num_levels=6,
        init_alpha=0.1,
        gate_bias=-2.0,
    ):
        super().__init__()
        self.video_dim = video_dim
        self.imu_dim = imu_dim
        if video_dim == embd_dim:
            self.video_residual = nn.Identity()
        else:
            self.video_residual = nn.Sequential(
                nn.Conv1d(video_dim, embd_dim, kernel_size=1),
                nn.GroupNorm(16, embd_dim),
                nn.ReLU(),
                nn.Conv1d(embd_dim, embd_dim, kernel_size=1),
                nn.GroupNorm(16, embd_dim),
                nn.ReLU(),
            )
        self.mravff = MRAVFFFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            n_head=n_head,
            num_levels=num_levels,
        )
        self.gate = nn.Conv1d(embd_dim * 2, embd_dim, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, video_feats, imu_feats, mask=None):
        assert video_feats.shape[1] == self.video_dim
        assert imu_feats.shape[1] == self.imu_dim
        video_base = self.video_residual(video_feats)
        mravff_out = self.mravff(video_feats, imu_feats, mask)
        gate = torch.sigmoid(self.gate(torch.cat([video_base, mravff_out], dim=1)))
        fused = video_base + self.alpha * gate * (mravff_out - video_base)
        if mask is not None:
            fused = fused * mask.to(fused.dtype)
        return fused


class IMUOnlyFusion(nn.Module):
    def __init__(self, video_dim, imu_dim, embd_dim, dropout=0.1):
        super().__init__()
        self.video_dim = video_dim
        self.imu_dim = imu_dim
        if imu_dim == embd_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv1d(imu_dim, embd_dim, kernel_size=1),
                nn.GroupNorm(16, embd_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(embd_dim, embd_dim, kernel_size=1),
                nn.GroupNorm(16, embd_dim),
                nn.ReLU(),
            )

    def forward(self, video_feats, imu_feats, mask=None):
        assert video_feats.shape[1] == self.video_dim
        assert imu_feats.shape[1] == self.imu_dim
        fused = self.proj(imu_feats)
        if mask is not None:
            fused = fused * mask.to(fused.dtype)
        return fused


def make_multimodal_fusion(fusion_type, video_dim, imu_dim, embd_dim, fusion_kwargs):
    fusion_type = fusion_type.lower()
    if fusion_type == "imu_only":
        return IMUOnlyFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    if fusion_type in ("mravff", "mravfffusion"):
        return MRAVFFFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    if fusion_type == "concat_project":
        return ConcatProjectFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    if fusion_type == "gate_sum":
        return GateSumFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    if fusion_type == "gate_concat":
        return GateConcatFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    if fusion_type == "residual_mravff":
        return ResidualMRAVFFFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    if fusion_type == "sensor_aware_gate_concat":
        return SensorAwareGateConcatFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    if fusion_type in ("ltca_gate_concat", "cross_attn_gate_concat"):
        return LocalTemporalCrossAttentionGateConcatFusion(
            video_dim=video_dim,
            imu_dim=imu_dim,
            embd_dim=embd_dim,
            **fusion_kwargs,
        )
    raise ValueError("Unknown fusion_type: {}".format(fusion_type))


@register_meta_arch("TinyHARFusionActionFormer")
class TinyHARFusionActionFormer(PtTransformer):
    def __init__(
        self,
        model_name,
        backbone_type,
        fpn_type,
        backbone_arch,
        scale_factor,
        input_dim,
        max_seq_len,
        max_buffer_len_factor,
        n_head,
        n_mha_win_size,
        embd_kernel_size,
        embd_dim,
        embd_with_ln,
        fpn_dim,
        fpn_with_ln,
        fpn_start_level,
        head_dim,
        regression_range,
        head_num_layers,
        head_kernel_size,
        head_with_ln,
        use_abs_pe,
        use_rel_pe,
        num_classes,
        train_cfg,
        test_cfg,
        video_dim=1024,
        imu_window_size=50,
        imu_input_dim=30,
        imu_project_dim=1024,
        fusion_dim=512,
        tinyhar_kwargs=None,
        fusion_kwargs=None,
    ):
        super().__init__(
            model_name=model_name,
            backbone_type=backbone_type,
            fpn_type=fpn_type,
            backbone_arch=backbone_arch,
            scale_factor=scale_factor,
            input_dim=fusion_dim,
            max_seq_len=max_seq_len,
            max_buffer_len_factor=max_buffer_len_factor,
            n_head=n_head,
            n_mha_win_size=n_mha_win_size,
            embd_kernel_size=embd_kernel_size,
            embd_dim=embd_dim,
            embd_with_ln=embd_with_ln,
            fpn_dim=fpn_dim,
            fpn_with_ln=fpn_with_ln,
            fpn_start_level=fpn_start_level,
            head_dim=head_dim,
            regression_range=regression_range,
            head_num_layers=head_num_layers,
            head_kernel_size=head_kernel_size,
            head_with_ln=head_with_ln,
            use_abs_pe=use_abs_pe,
            use_rel_pe=use_rel_pe,
            num_classes=num_classes,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if tinyhar_kwargs is None:
            tinyhar_kwargs = {}
        if fusion_kwargs is None:
            fusion_kwargs = {}
        self.expected_video_dim = video_dim
        self.expected_imu_dim = imu_window_size * imu_input_dim
        self.tinyhar_encoder = TinyHAREncoder(
            input_dim=imu_input_dim,
            window_size=imu_window_size,
            project_dim=imu_project_dim,
            projection_dropout=tinyhar_kwargs.get("projection_dropout", 0.1),
            tinyhar_kwargs=tinyhar_kwargs,
        )
        self.fusion_module = MRAVFFFusion(
            video_dim=video_dim,
            imu_dim=imu_project_dim,
            embd_dim=fusion_dim,
            **fusion_kwargs,
        )

    def preprocessing(self, video_list, padding_val=0.0):
        feats = [x["feats"] for x in video_list]
        imu_feats = [x["additional_feats"] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert (
                max_len <= self.max_seq_len
            ), "Input length must be smaller than max_seq_len during training"
            max_len = self.max_seq_len
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        batch_size = len(feats)
        video_batch = feats[0].new_full(
            (batch_size, self.expected_video_dim, max_len), padding_val
        )
        imu_batch = imu_feats[0].new_full(
            (batch_size, self.expected_imu_dim, max_len), padding_val
        )
        for idx, (video_feat, imu_feat) in enumerate(zip(feats, imu_feats)):
            assert video_feat.shape[0] == self.expected_video_dim
            assert imu_feat.shape[0] == self.expected_imu_dim
            assert video_feat.shape[-1] == imu_feat.shape[-1]
            cur_len = video_feat.shape[-1]
            video_batch[idx, :, :cur_len].copy_(video_feat)
            imu_batch[idx, :, :cur_len].copy_(imu_feat)

        mask = torch.arange(max_len)[None, :] < feats_lens[:, None]
        video_batch = video_batch.to(self.device)
        imu_batch = imu_batch.to(self.device)
        mask = mask.unsqueeze(1).to(self.device)

        imu_encoded = self.tinyhar_encoder(imu_batch)
        fused = self.fusion_module(video_batch, imu_encoded, mask)
        return fused, mask


@register_meta_arch("MultiModalActionFormer")
class MultiModalActionFormer(PtTransformer):
    def __init__(
        self,
        model_name,
        backbone_type,
        fpn_type,
        backbone_arch,
        scale_factor,
        input_dim,
        max_seq_len,
        max_buffer_len_factor,
        n_head,
        n_mha_win_size,
        embd_kernel_size,
        embd_dim,
        embd_with_ln,
        fpn_dim,
        fpn_with_ln,
        fpn_start_level,
        head_dim,
        regression_range,
        head_num_layers,
        head_kernel_size,
        head_with_ln,
        use_abs_pe,
        use_rel_pe,
        num_classes,
        train_cfg,
        test_cfg,
        video_dim=1024,
        imu_window_size=50,
        imu_input_dim=30,
        imu_project_dim=1024,
        raw_imu_project_dim=1024,
        fusion_dim=512,
        imu_encoder_type="tinyhar",
        fusion_type="mravff",
        tinyhar_kwargs=None,
        fusion_kwargs=None,
    ):
        super().__init__(
            model_name=model_name,
            backbone_type=backbone_type,
            fpn_type=fpn_type,
            backbone_arch=backbone_arch,
            scale_factor=scale_factor,
            input_dim=fusion_dim,
            max_seq_len=max_seq_len,
            max_buffer_len_factor=max_buffer_len_factor,
            n_head=n_head,
            n_mha_win_size=n_mha_win_size,
            embd_kernel_size=embd_kernel_size,
            embd_dim=embd_dim,
            embd_with_ln=embd_with_ln,
            fpn_dim=fpn_dim,
            fpn_with_ln=fpn_with_ln,
            fpn_start_level=fpn_start_level,
            head_dim=head_dim,
            regression_range=regression_range,
            head_num_layers=head_num_layers,
            head_kernel_size=head_kernel_size,
            head_with_ln=head_with_ln,
            use_abs_pe=use_abs_pe,
            use_rel_pe=use_rel_pe,
            num_classes=num_classes,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if tinyhar_kwargs is None:
            tinyhar_kwargs = {}
        if fusion_kwargs is None:
            fusion_kwargs = {}

        self.expected_video_dim = int(video_dim)
        self.expected_imu_dim = int(imu_window_size) * int(imu_input_dim)
        self.imu_encoder_type = imu_encoder_type.lower()
        self.fusion_type = fusion_type.lower()

        if self.imu_encoder_type == "tinyhar":
            self.imu_encoder = TinyHAREncoder(
                input_dim=imu_input_dim,
                window_size=imu_window_size,
                project_dim=imu_project_dim,
                projection_dropout=tinyhar_kwargs.get("projection_dropout", 0.1),
                tinyhar_kwargs=tinyhar_kwargs,
            )
            self.encoded_imu_dim = int(imu_project_dim)
        elif self.imu_encoder_type in ("raw1024", "none_raw1024"):
            self.imu_encoder = None
            self.encoded_imu_dim = int(raw_imu_project_dim)
            raw_indices = torch.round(
                torch.linspace(0, self.expected_imu_dim - 1, self.encoded_imu_dim)
            ).long()
            self.register_buffer("raw_imu_indices", raw_indices, persistent=False)
        elif self.imu_encoder_type in ("raw", "none_raw", "raw1500"):
            self.imu_encoder = None
            self.encoded_imu_dim = self.expected_imu_dim
            self.register_buffer(
                "raw_imu_indices",
                torch.arange(self.expected_imu_dim).long(),
                persistent=False,
            )
        else:
            raise ValueError("Unknown imu_encoder_type: {}".format(imu_encoder_type))

        self.fusion_module = make_multimodal_fusion(
            self.fusion_type,
            video_dim=self.expected_video_dim,
            imu_dim=self.encoded_imu_dim,
            embd_dim=fusion_dim,
            fusion_kwargs=fusion_kwargs,
        )

    def _encode_imu(self, imu_batch):
        if self.imu_encoder_type == "tinyhar":
            return self.imu_encoder(imu_batch)
        return imu_batch.index_select(1, self.raw_imu_indices.to(imu_batch.device))

    def preprocessing(self, video_list, padding_val=0.0):
        feats = [x["feats"] for x in video_list]
        imu_feats = [x["additional_feats"] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert (
                max_len <= self.max_seq_len
            ), "Input length must be smaller than max_seq_len during training"
            max_len = self.max_seq_len
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        batch_size = len(feats)
        video_batch = feats[0].new_full(
            (batch_size, self.expected_video_dim, max_len), padding_val
        )
        imu_batch = imu_feats[0].new_full(
            (batch_size, self.expected_imu_dim, max_len), padding_val
        )
        for idx, (video_feat, imu_feat) in enumerate(zip(feats, imu_feats)):
            assert video_feat.shape[0] == self.expected_video_dim
            assert imu_feat.shape[0] == self.expected_imu_dim
            assert video_feat.shape[-1] == imu_feat.shape[-1]
            cur_len = video_feat.shape[-1]
            video_batch[idx, :, :cur_len].copy_(video_feat)
            imu_batch[idx, :, :cur_len].copy_(imu_feat)

        mask = torch.arange(max_len)[None, :] < feats_lens[:, None]
        video_batch = video_batch.to(self.device)
        imu_batch = imu_batch.to(self.device)
        mask = mask.unsqueeze(1).to(self.device)

        imu_encoded = self._encode_imu(imu_batch)
        fused = self.fusion_module(video_batch, imu_encoded, mask)
        return fused, mask
