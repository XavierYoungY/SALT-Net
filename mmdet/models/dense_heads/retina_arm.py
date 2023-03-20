import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init, Scale

from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmcv.ops import DeformConv2d
import numpy as np
import torch
from mmdet.core import distance2bbox, multi_apply, images_to_levels
from mmcv.runner import force_fp32



@HEADS.register_module()
class NFHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = NFHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_bbox_init=dict(type='L1Loss', loss_weight=0.5),
                 num_points=9,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.num_points = num_points

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        super(NFHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)
        self.loss_bbox_init = build_loss(loss_bbox_init)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

        #coarse bbox prediction dimensions
        self.init_out_dim = 2 * self.num_points
        #...........................Classification
        self.ram_cls_refinement = ProposalConvModule(self.feat_channels,
                                                     self.feat_channels,
                                                     self.init_out_dim)
        self.ram_cls_conv = DCNConvModule(self.feat_channels,
                                          self.feat_channels, self.dcn_kernel,
                                          1, self.dcn_pad)

        #...........................Regression
        self.reg_init_out = RProposalConvModule(self.feat_channels,
                                                self.feat_channels,
                                                self.init_out_dim - 4)
        self.ram_reg_conv = DCNConvModule(self.feat_channels,
                                          self.feat_channels, self.dcn_kernel,
                                          1, self.dcn_pad)
        self.scales_init = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
        normal_init(self.ram_cls_conv.conv, std=0.01)
        normal_init(self.ram_reg_conv.conv, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales_init)

    def forward_single(self, x, scale_init):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_feat = cls_feat + x

        #regression..........................
        reg_init, scale_fs = self.reg_init_out(reg_feat)
        bbox_distance = scale_init(reg_init).float()
        reg_points = self.points_sampler(bbox_distance, scale_fs)
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        reg_dcn_offset = reg_points - dcn_base_offset

        reg_feat = self.ram_reg_conv(reg_feat, reg_dcn_offset)

        bbox_pred = self.retina_reg(reg_feat)
        #classification ...........................................
        class_offset = (self.ram_cls_refinement(cls_feat)
                        ).exp() * reg_points.detach() - dcn_base_offset

        cls_feat = self.ram_cls_conv(cls_feat, class_offset)
        cls_score = self.retina_cls(cls_feat)

        if self.training:
            return cls_score, bbox_pred, bbox_distance
        else:
            return cls_score, bbox_pred

    def points_sampler(self, bbox, scale):
        # bbox: distance to left top right bottom
        #TODO: y_fisrt for points 第一列是y
        '''
        0|1|2
        3|4|5
        6|7|8
        EXtreme points are:left->3; top->1; right->5; bottom->7
        scales: y_fisrt 第一列是y 前四个是 left top right bottom 其它方向的尺度: y x y x

        '''
        N, C, H, W = bbox.size()
        width = bbox[:, [0], ...] + bbox[:, [2], ...]
        height = bbox[:, [1], ...] + bbox[:, [3], ...]
        xmin = -bbox[:, [0], ...]
        ymin = -bbox[:, [1], ...]
        xmax = bbox[:, [2], ...]
        ymax = bbox[:, [3], ...]
        pts = bbox.new_zeros(N, self.init_out_dim, H, W)
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        #前四个点是left top right bottom
        pts_reshape[:, [3], 1, ...] = xmin
        pts_reshape[:, [5], 1, ...] = xmax
        pts_reshape[:, [3, 5], 0, ...] = height * scale[:, [0, 2], ...] + ymin
        pts_reshape[:, [1, 7], 1, ...] = width * scale[:, [1, 3], ...] + xmin
        pts_reshape[:, [1], 0, ...] = ymin
        pts_reshape[:, [7], 0, ...] = ymax

        pts_reshape[:, [0, 2, 4, 6, 8], 1,
                    ...] = xmin + width * scale[:, [5, 7, 9, 11, 13], ...]
        pts_reshape[:, [0, 2, 4, 6, 8], 0,
                    ...] = ymin + height * scale[:, [4, 6, 8, 10, 12], ...]

        return pts_reshape.view(*pts.shape)


    def loss_single(self, cls_score, bbox_pred, bbox_distance, anchors, labels,
                    label_weights, bbox_targets, bbox_weights,
                    num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_distance = bbox_distance.permute(0, 2, 3, 1).reshape(-1, 4)

        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_distance = self.bbox_coder.decode(anchors, bbox_distance)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        loss_bbox_init = self.loss_bbox_init(
            bbox_distance,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_bbox_init

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_distances'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_distances,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_bbox_init = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            bbox_distances,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_bbox_init=losses_bbox_init)


class DCNConvModule(nn.Module):

    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        stride=1,
        dcn_pad=1,
        num_groups=32,
    ):
        super(DCNConvModule, self).__init__()

        self.conv = DeformConv2d(in_channels, out_channels, kernel_size,
                                 stride, dcn_pad)
        self.bn = nn.GroupNorm(num_groups, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, offset):

        return self.relu(self.bn(self.conv(x, offset)))


class ProposalConvModule(nn.Module):

    def __init__(
        self,
        in_channels=256,
        inter_channels=256,
        out_channels=18,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=32,
    ):
        super(ProposalConvModule, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.proposal = nn.Conv2d(
            inter_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.GroupNorm(num_groups, inter_channels)
        self.relu = nn.ReLU(inplace=True)

        self.initialize()

    def initialize(self):
        normal_init(self.conv, std=0.01)
        normal_init(self.proposal, std=0.01)

    def forward(self, x):

        return self.proposal(self.relu(self.bn(self.conv(x))))


class RProposalConvModule(nn.Module):

    def __init__(
        self,
        in_channels=256,
        inter_channels=256,
        out_channels=14,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=32,
    ):
        super(RProposalConvModule, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.regression = nn.Conv2d(
            inter_channels,
            4,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.scale_fs = nn.Conv2d(
            inter_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn = nn.GroupNorm(num_groups, inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.initialize()

    def initialize(self):
        normal_init(self.conv, std=0.01)
        normal_init(self.regression, std=0.01)
        normal_init(self.scale_fs, std=0.01)

    def forward(self, x):
        feats = self.relu(self.bn(self.conv(x)))
        #当前位置与四个边的距离
        bbox_distance = self.relu(self.regression(feats))
        scale_fs = self.sigmoid(self.scale_fs(feats))
        return bbox_distance, scale_fs
