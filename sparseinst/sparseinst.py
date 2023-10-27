# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmdet.models import BaseDetector
from mmseg.models.segmentors import BaseSegmentor
#from mmdet.models.utils import unpack_gt_instances
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType,SampleList,OptSampleList
from mmseg.models.utils import resize
from mmengine.structures import PixelData
@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) /
                     (mask_pred_.sum([1, 2]) + 1e-6))


@MODELS.register_module()
class SparseInst(BaseDetector):
    """Implementation of `SparseInst <https://arxiv.org/abs/1912.02424>`_

    Args:
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        encoder (:obj:`ConfigDict` or dict): The encoder module.
        decoder (:obj:`ConfigDict` or dict): The decoder module.
        criterion (:obj:`ConfigDict` or dict, optional): The training matcher
            and losses. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of SparseInst. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: ConfigType,
                 backbone: ConfigType,
                 encoder: ConfigType,
                 decoder: ConfigType,
                 criterion: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # backbone
        self.backbone = MODELS.build(backbone)
        # encoder & decoder
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

        # matcher & loss (matcher is built in loss)
        self.criterion = MODELS.build(criterion)

        # inference
        self.cls_threshold = test_cfg.score_thr
        self.mask_threshold = test_cfg.mask_thr_binary
        self.ignore_index = 255

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_MaskFormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []
        for data_sample in batch_data_samples:
            # Add `batch_input_shape` in metainfo of data_sample, which would
            # be used in MaskFormerHead of MMDetection.
            metainfo = data_sample.metainfo
            metainfo['batch_input_shape'] = metainfo['img_shape']
            data_sample.set_metainfo(metainfo)
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros((0, gt_sem_seg.shape[-2],
                                        gt_sem_seg.shape[-1])).to(gt_sem_seg)
            else:
                gt_masks = torch.stack(masks).squeeze(1)

            instance_data = InstanceData(
                labels=gt_labels, masks=gt_masks.long())
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas
    
    
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.backbone(batch_inputs)
        x = self.encoder(x)
        results = self.decoder(x)
        return results

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        max_shape = batch_inputs.shape[-2:]
        output = self._forward(batch_inputs)

        pred_scores = output['pred_logits'].sigmoid()
        pred_masks = output['pred_masks'].sigmoid()
        pred_objectness = output['pred_scores'].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        seg_logits = []
        for batch_idx, (scores_per_image, mask_pred_per_image,
                        datasample) in enumerate(
                            zip(pred_scores, pred_masks, batch_data_samples)):
            result = InstanceData()
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]
            scores_per_image = scores_per_image[keep]
            if scores.size(0) == 0:
                result.scores = scores
                result.labels = labels
                results_list.append(result)
                continue

            img_meta = datasample.metainfo
            # rescoring mask using maskness
            scores = rescoring_mask(scores,
                                    mask_pred_per_image > self.mask_threshold,
                                    mask_pred_per_image)
            h, w = img_meta['img_shape'][:2]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image.unsqueeze(1),
                size=max_shape,
                mode='bilinear',
                align_corners=False)[:, :, :h, :w]

            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                mask_pred_per_image = F.interpolate(
                    mask_pred_per_image,
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            
            scores_per_image = F.softmax(scores_per_image, dim=-1)[..., :-1]
            semseg = torch.einsum("qc,qhw->chw", scores_per_image, mask_pred_per_image)
            seg_logits.append(semseg)
            
        seg_logits = torch.stack(seg_logits, dim=0)
        return self.postprocess_result(seg_logits, batch_data_samples)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self._forward(batch_inputs)
        #(batch_gt_instances, batch_gt_instances_ignore,
        # batch_img_metas) = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)
        losses = self.criterion(outs, batch_gt_instances, batch_img_metas)
        return losses

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        x = self.encoder(x)
        return x
    
    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=False,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples

