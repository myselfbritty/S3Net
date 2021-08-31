import torch
import torch.nn as nn
import numpy as np
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
import pycocotools._mask as _mask
import pycocotools.mask as maskUtils

@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    ##Computer overlap between the segms
    def compute_mask_IU(self, mask_gt, mask_pred):
        assert(mask_pred.shape[-2:] == mask_gt.shape[-2:])
        temp = (mask_gt * mask_pred)
        intersection = temp.sum()
        union = ((mask_gt + mask_pred) - temp).sum()
        return intersection, union
    ####Heuristic added to remove overlapping segms)
    def remove_overlapping_segms(self, segm_results, det_bboxes, det_labels, iou_thresh, score_thresh):
        cand_segm = segm_results.copy()
        new_segm_results = {}
        non_maximum_cands_vector = []
        new_det_bboxes = []
        new_det_labels = []
        segm_masks = []
        indices_det = []
        temp = []
        ## Segm_masks are in classes format, flattening to candidates according to detection bboxes
        for index, s in enumerate(det_labels):
            segm_mask = maskUtils.decode(cand_segm[s])
            exact_classes = set(det_labels.tolist())

            if ((segm_mask.shape[2:][0] > 1) and (s not in temp)):
                temp.append(s)
                indices = [j for j, element in enumerate(det_labels) if element == s]

                for itr in range(len(indices)):
                    indices_det.append(indices[itr])
                for it_s in range(segm_mask.shape[2:][0]):
                    new_segm_mask = segm_mask[:, :, it_s, np.newaxis]
                    new_segm_mask[new_segm_mask==1] = 255
                    segm_masks.append(new_segm_mask)
            elif (s in temp):
                continue
            else:
                segm_mask[segm_mask==1] = 255
                indices_det.append(index)
                segm_masks.append(segm_mask)
        for iterat in range(len(segm_masks)):
            in_segm_mask = segm_masks[iterat]
            overlaped_cands = [indices_det[iterat]]
            cands_scores = [det_bboxes[indices_det[iterat]].tolist()[4]]
            for iterat_next in range(len(segm_masks)-iterat-1):
                other_cand_segm_mask = segm_masks[iterat+iterat_next+1]
                i, u = self.compute_mask_IU(in_segm_mask/255, other_cand_segm_mask/255)
                if (i + 1e-15)/(u + 1e-15) > iou_thresh:
                    overlaped_cands.append(indices_det[iterat+iterat_next+1])
                    cands_scores.append(det_bboxes[indices_det[iterat+iterat_next+1]].tolist()[4])
            if len(overlaped_cands) > 1:
                cands_scores_ar = np.array(cands_scores, dtype=float)
                max_score_idx = np.argmax(cands_scores_ar)
                overlaped_cands.pop(max_score_idx)
                non_maximum_cands = overlaped_cands
                for nm_cand in non_maximum_cands:
                    non_maximum_cands_vector.append(nm_cand)
        to_be_removed = set(non_maximum_cands_vector)
        count = 0
        for c, cand in enumerate(det_bboxes.tolist()):
            if c in to_be_removed:
                dummy_variable = 0
            else:
                if det_bboxes[c].tolist()[4] > score_thresh:
                    new_det_bboxes.append(det_bboxes.tolist()[c])
                    new_det_labels.append(det_labels.tolist()[c])
                    count +=1
        det_bboxes = torch.tensor(new_det_bboxes).cuda()
        det_labels = torch.tensor(new_det_labels).cuda()
        
        return det_bboxes, det_labels


    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        
        segm_results = self.simple_test_mask(
            x, img_meta, det_bboxes, det_labels, rescale=rescale)
        iou_thresh = 0.5 # to find overlapping segms
        score_thresh = 0.1 # to remove ambiguous boxes
        det_bboxes, det_labels = self.remove_overlapping_segms(segm_results, det_bboxes, det_labels, 0.5, 0.1)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
