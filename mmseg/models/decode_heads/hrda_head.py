from copy import deepcopy

from abc import ABC
import torch.nn as nn

import torch
from torch.nn import functional as F

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead


def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.1
        self.base_temperature = 0.07

        self.ignore_label = -1

        self.max_samples = 1024
        self.max_views = 100

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss

@HEADS.register_module()
class HRDAHead(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention

        # Contrast Loss Add
        self.contrast_criterion = PixelContrastLoss()
        self.loss_weight = 0.1

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_logits = self.head(features)
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        lr_seg = self.head(lr_inp)
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')

        hr_seg = self.decode_hr(hr_inp, batch_size)

        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg

        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      feature_outputs,
                      stage,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        if stage == 1:
            # Contrast Loss
            target = torch.squeeze(gt_semantic_seg, dim=1)
            h, w = target.size(1), target.size(2)

            seg = feature_outputs['seg']
            embedding = feature_outputs['embed']

            _, predict = torch.max(seg, 1)
            loss_contrast = self.contrast_criterion(embedding, target, predict)
            losses['loss_seg'] = losses['loss_seg'] + self.loss_weight * loss_contrast

        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        fused_seg, lr_seg, hr_seg = seg_logit
        loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            self.debug_output['Cropped GT'] = \
                cropped_seg_label.squeeze(1).detach().cpu().numpy()
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        return loss
