import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        point2bbox, images_to_levels, multi_apply,
                        multiclass_nms, reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmcv.ops import DeformConv2d
import numpy as np


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@HEADS.register_module()
class SALTHead(AnchorHead):
    """
    Example:
        >>> self = SALTHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 loss_bbox_init=dict(type='GIoULoss', loss_weight=1.0),
                 reg_max=16,
                 reg_topk=4,
                 num_points=9,
                 reg_channels=64,
                 add_mean=True,
                 dcn_on_last_conv=False,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.num_points = num_points
        self.dcn_on_last_conv = dcn_on_last_conv
        if add_mean:
            self.total_dim += 1
        print('total dim = ', self.total_dim * 4)

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

        super(SALTHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)
        self.loss_bbox_init = build_loss(loss_bbox_init)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        #coarse bbox prediction dimensions
        self.init_out_dim = 2 * self.num_points
        #...........................Classification
        self.salt_cls_refinement = ProposalConvModule(self.feat_channels,
                                                     self.feat_channels,
                                                     self.init_out_dim)
        self.salt_cls_conv = DCNConvModule(self.feat_channels,
                                          self.feat_channels, self.dcn_kernel,
                                          1, self.dcn_pad)

        self.salt_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        #...........................Regression
        self.reg_init_out = RProposalConvModule(self.feat_channels,
                                                self.feat_channels,
                                                self.init_out_dim - 4)
        self.salt_reg_conv = DCNConvModule(self.feat_channels,
                                          self.feat_channels, self.dcn_kernel,
                                          1, self.dcn_pad)
        self.salt_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)

        self.scales_init = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]

        self.reg_conf = nn.Sequential(*conf_vector)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_conf:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.salt_cls_conv.conv, std=0.01)
        normal_init(self.salt_cls, std=0.01, bias=bias_cls)
        normal_init(self.salt_reg_conv.conv, std=0.01)
        normal_init(self.salt_reg, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.scales_init)

    def forward_single(self, x, scale, scale_init):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_feat = cls_feat + x

        #regression init : the displacement vectors
        reg_init, scale_fs = self.reg_init_out(reg_feat)
        bbox_distance = scale_init(reg_init).float()
        reg_points = self.points_sampler(bbox_distance, scale_fs)

        dcn_base_offset = self.dcn_base_offset.type_as(x)
       
        reg_dcn_offset = reg_points - dcn_base_offset

        bbox_pred = scale(
            self.salt_reg(self.salt_reg_conv(reg_feat, reg_dcn_offset))).float()

        N, C, H, W = bbox_pred.size()  # batch 17*4 H W
        #selecting the top-k
        prob = F.softmax(
            bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)
        #top k bbox predictions + mean
        if self.add_mean:
            stat = torch.cat(
                [prob_topk, prob_topk.mean(dim=2, keepdim=True)], dim=2)
        else:
            stat = prob_topk
        #2x 1x1 conv and sigmoid
        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))

        #classification ...........................................
        class_offset = (self.salt_cls_refinement(cls_feat)
                        ).exp() * reg_points.detach() - dcn_base_offset

        cls_feat = self.salt_cls_conv(cls_feat, class_offset)

        cls_score = self.salt_cls(cls_feat).sigmoid() * quality_score
        if self.training:
            return cls_score, bbox_pred, bbox_distance
        else:
            return cls_score, bbox_pred

    def points_sampler(self, bbox, scale):
        # bbox: distance to left top right bottom
        #TODO: y_fisrt for points 
        '''
        0|1|2
        3|4|5
        6|7|8
        EXtreme points are:left->3; top->1; right->5; bottom->7
        scales: left top right bottom 
        y_fisrt : y x y x

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
        #first 4 points:left top right bottom
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

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)


    def loss_single(self, anchors, cls_score, bbox_pred, bbox_distance, labels,
                    label_weights, bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.

        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))

        bbox_distance = bbox_distance.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_reg_init = bbox_distance[pos_inds]
            #distance to the four bounds
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            #init :relative to bbox
            pos_reg_init_bbox = distance2bbox(pos_anchor_centers, pos_reg_init)
            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            #distance to the four bounds
            #init + DCN 
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            #xmin ymin xmax ymax
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            #[N*4,16+1]
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            #[N*4]
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            # regression init loss 
            loss_bbox_init = self.loss_bbox_init(
                pos_reg_init_bbox,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)
            # print(loss_bbox_init)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox_init = bbox_distance.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0, device=bbox_pred[0].device)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum(
        ), loss_bbox_init

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
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # feature size
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # ATSS
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

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl,\
            avg_factor, losses_bbox_init = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                bbox_distances,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        losses_bbox_init = list(
            map(lambda x: x / avg_factor, losses_bbox_init))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_bbox_init=losses_bbox_init)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                has shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for a single
                scale level with shape (4*(n+1), H, W), n is max value of
                integral set.
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (N,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, anchors in zip(
                cls_scores, bbox_preds, self.anchor_generator.strides,
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2,
                                       0).reshape(-1, self.cls_out_channels)
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for oat head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside


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
        #distance to the four bounds
        bbox_distance = self.relu(self.regression(feats))
        scale_fs = self.sigmoid(self.scale_fs(feats))
        return bbox_distance, scale_fs
