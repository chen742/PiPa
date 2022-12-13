# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
#
# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import dataloaders
import json
from itertools import cycle

import torch.nn.functional as F
from torch import nn


import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        # CAC
        self.out_dim = 19
        self.proj_final_dim = 128
        self.project = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dim, self.proj_final_dim, kernel_size=1, stride=1)
        )

        self.classifier = nn.Conv2d(self.out_dim, 19, kernel_size=1, stride=1)

        self.weight_unsup = 0.01
        self.temp = 0.1
        self.epoch_start_unsup = 0
        self.selected_num = 1600
        self.step_save = 2
        self.step_count = 0
        self.feature_bank = []
        self.pseudo_label_bank = []
        self.pos_thresh_value = 0.9
        self.stride = 32

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                for _ in range(1)]
            tensors_gather = [tensor]

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        if self.local_iter % self.debug_img_interval == 0:
            self.get_model().decode_head.debug = True
            self.get_ema_model().decode_head.debug = True
        else:
            self.get_model().decode_head.debug = False
            self.get_ema_model().decode_head.debug = False

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Source stage: 1
        stage = 1
        clean_losses = self.get_model().forward_train(
            stage, img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().decode_head.debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().generate_pseudo_label(
            target_img, target_img_metas)
        seg_debug['Target'] = self.get_ema_model().decode_head.debug_output

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        del ema_logits
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)
        del pseudo_prob, ps_large_p, ps_size

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        del gt_pixel_weight
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Tatget (Mix) stage: 2
        stage = 2
        mix_losses = self.get_model().forward_train(
            stage, mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=False)
        seg_debug['Mix'] = self.get_model().decode_head.debug_output
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        if self.local_iter < 10000: # warm up iterations
            mix_loss.backward()
        else:
            # DATA_LOADER 2
            def init_dl():
                # LOAD Config
                config = json.load(open('configs/cac/config.json'))

                # DATA LOADERS
                config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
                config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
                config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
                config['train_supervised']['data_dir'] = config['data_dir']
                config['train_unsupervised']['data_dir'] = config['data_dir']
                config['val_loader']['data_dir'] = config['data_dir']
                config['train_supervised']['datalist'] = config['datalist']
                config['train_unsupervised']['datalist'] = config['datalist']
                config['val_loader']['datalist'] = config['datalist']

                if config['dataset'] == 'cityscapes':
                    sup_dataloader = dataloaders.City
                    unsup_dataloader = dataloaders.PairCity

                supervised_loader = sup_dataloader(config['train_supervised'])
                unsupervised_loader = unsup_dataloader(config['train_unsupervised'])
                self.supervised_loader = supervised_loader
                self.unsupervised_loader = unsupervised_loader

                self.dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            a = 1
            if a == 1:
                init_dl()
                a += 1

            (input_l, target_l), (input_ul, target_ul, ul1, br1, ul2, br2, flip) = next(self.dataloader)
            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
            kargs = {'gpu': 0, 'ul1': ul1, 'br1': br1, 'ul2': ul2, 'br2': br2, 'flip': flip}

            # Params into the model
            xl = input_l
            target_l = target_l
            x_ul = input_ul
            target_ul = target_ul
            x_ul1 = x_ul[:, 0, :, :, :]
            x_ul2 = x_ul[:, 1, :, :, :]

            fuse_xul1 = self.get_ema_model().encode_decode(
                x_ul1, target_img_metas)
            enc_ul1 = fuse_xul1
            # Downsample
            enc_ul1 = F.avg_pool2d(enc_ul1, kernel_size=2, stride=2)
            output_ul1 = self.project(enc_ul1)  # [b, c, h, w]
            output_ul1 = F.normalize(output_ul1, 2, 1)
            # Process xul2
            fuse_xul2 = self.get_ema_model().encode_decode(
                x_ul2, target_img_metas)
            enc_ul2 = fuse_xul2
            # Downsample
            enc_ul2 = F.avg_pool2d(enc_ul2, kernel_size=2, stride=2)
            output_ul2 = self.project(enc_ul2)  # [b, c, h, w]
            output_ul2 = F.normalize(output_ul2, 2, 1)
            # compute pseudo label
            logits1 = self.classifier(enc_ul1)  # [batch_size, num_classes, h, w]
            logits2 = self.classifier(enc_ul2)
            pseudo_logits_1 = F.softmax(logits1, 1).max(1)[0].detach()  # [batch_size, h, w]
            pseudo_logits_2 = F.softmax(logits2, 1).max(1)[0].detach()
            pseudo_label1 = logits1.max(1)[1].detach()  # [batch_size, h, w]
            pseudo_label2 = logits2.max(1)[1].detach()
            # get overlap part
            output_feature_list1 = []
            output_feature_list2 = []
            pseudo_label_list1 = []
            pseudo_label_list2 = []
            pseudo_logits_list1 = []
            pseudo_logits_list2 = []
            for idx in range(x_ul1.size(0)):
                output_ul1_idx = output_ul1[idx]
                output_ul2_idx = output_ul2[idx]
                pseudo_label1_idx = pseudo_label1[idx]
                pseudo_label2_idx = pseudo_label2[idx]
                pseudo_logits_1_idx = pseudo_logits_1[idx]
                pseudo_logits_2_idx = pseudo_logits_2[idx]
                if flip[0][idx] == True:
                    output_ul1_idx = torch.flip(output_ul1_idx, dims=(2,))
                    pseudo_label1_idx = torch.flip(pseudo_label1_idx, dims=(1,))
                    pseudo_logits_1_idx = torch.flip(pseudo_logits_1_idx, dims=(1,))
                if flip[1][idx] == True:
                    output_ul2_idx = torch.flip(output_ul2_idx, dims=(2,))
                    pseudo_label2_idx = torch.flip(pseudo_label2_idx, dims=(1,))
                    pseudo_logits_2_idx = torch.flip(pseudo_logits_2_idx, dims=(1,))
                output_feature_list1.append(
                    output_ul1_idx[:, ul1[0][idx] // 8:br1[0][idx] // 8, ul1[1][idx] // 8:br1[1][idx] // 8].permute(1, 2,
                                                                                                                    0).contiguous().view(
                        -1, output_ul1.size(1)))
                output_feature_list2.append(
                    output_ul2_idx[:, ul2[0][idx] // 8:br2[0][idx] // 8, ul2[1][idx] // 8:br2[1][idx] // 8].permute(1, 2,
                                                                                                                    0).contiguous().view(
                        -1, output_ul2.size(1)))
                pseudo_label_list1.append(pseudo_label1_idx[ul1[0][idx] // 8:br1[0][idx] // 8,
                                          ul1[1][idx] // 8:br1[1][idx] // 8].contiguous().view(-1))
                pseudo_label_list2.append(pseudo_label2_idx[ul2[0][idx] // 8:br2[0][idx] // 8,
                                          ul2[1][idx] // 8:br2[1][idx] // 8].contiguous().view(-1))
                pseudo_logits_list1.append(pseudo_logits_1_idx[ul1[0][idx] // 8:br1[0][idx] // 8,
                                           ul1[1][idx] // 8:br1[1][idx] // 8].contiguous().view(-1))
                pseudo_logits_list2.append(pseudo_logits_2_idx[ul2[0][idx] // 8:br2[0][idx] // 8,
                                           ul2[1][idx] // 8:br2[1][idx] // 8].contiguous().view(-1))
            output_feat1 = torch.cat(output_feature_list1, 0)  # [n, c]
            output_feat2 = torch.cat(output_feature_list2, 0)  # [n, c]
            pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0)  # [n,]
            pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0)  # [n,]
            pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0)  # [n,]
            pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0)  # [n,]
            assert output_feat1.size(0) == output_feat2.size(0)
            assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
            assert output_feat1.size(0) == pseudo_label1_overlap.size(0)

            # concat across multi-gpus
            b, c, h, w = output_ul1.size()
            selected_num = self.selected_num
            output_ul1_flatten = output_ul1.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
            output_ul2_flatten = output_ul2.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
            selected_idx1 = np.random.choice(range(b * h * w), selected_num, replace=False)
            selected_idx2 = np.random.choice(range(b * h * w), selected_num, replace=False)
            output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
            output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
            output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected],
                                                   0)  # [2*kk, c]

            output_ul_all = self.concat_all_gather(output_ul_flatten_selected)

            pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
            pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
            pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected],
                                                      0)  # [2*kk]

            pseudo_label_all = self.concat_all_gather(pseudo_label_flatten_selected)

            self.feature_bank.append(output_ul_all)
            self.pseudo_label_bank.append(pseudo_label_all)
            if self.step_count > self.step_save:
                self.feature_bank = self.feature_bank[1:]
                self.pseudo_label_bank = self.pseudo_label_bank[1:]
            else:
                self.step_count += 1
            output_ul_all = torch.cat(self.feature_bank, 0)
            pseudo_label_all = torch.cat(self.pseudo_label_bank, 0)
            eps = 1e-8
            pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp  # [n, 1]
            pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp  # [n, 1]

            # compute loss1
            b = 8000
            def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
                # print("gpu: {}, i_1: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float()  # [n, b]
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp  # [n, b]
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
                return logits1_neg_idx

            def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
                # print("gpu: {}, i_1_0: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float()  # [n, b]
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp  # [n, b]
                neg1_idx = torch.cat([pos, neg1_idx], 1)  # [n, 1+b]
                mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).float().cuda(), mask1_idx], 1)  # [n, 1+b]
                neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0]  # [n, 1]
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
                return logits1_neg_idx, neg_max1

            N = output_ul_all.size(0)
            logits1_down = torch.zeros(pos1.size(0)).float().cuda()
            for i in range((N - 1) // b + 1):
                # print("gpu: {}, i: {}".format(gpu, i))
                pseudo_label_idx = pseudo_label_all[i * b:(i + 1) * b]
                output_ul_idx = output_ul_all[i * b:(i + 1) * b]
                if i == 0:
                    logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1, output_ul_idx,
                                                                                  pseudo_label_idx, pseudo_label1_overlap)
                else:
                    logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx,
                                                                        pseudo_label_idx, pseudo_label1_overlap, neg_max1)
                logits1_down += logits1_neg_idx

            logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)
            pos_mask_1 = ((pseudo_logits2_overlap > self.pos_thresh_value) & (
                    pseudo_logits1_overlap < pseudo_logits2_overlap)).float()
            loss1 = -torch.log(logits1 + eps)
            loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)

            # compute loss2
            def run2(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2):
                # print("gpu: {}, i_2: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float()  # [n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp  # [n, b]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1)  # [n, ]
                return logits2_neg_idx

            def run2_0(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap):
                # print("gpu: {}, i_2_0: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float()  # [n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp  # [n, b]
                neg2_idx = torch.cat([pos, neg2_idx], 1)  # [n, 1+b]
                mask2_idx = torch.cat([torch.ones(mask2_idx.size(0), 1).float().cuda(), mask2_idx], 1)  # [n, 1+b]
                neg_max2 = torch.max(neg2_idx, 1, keepdim=True)[0]  # [n, 1]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1)  # [n, ]
                return logits2_neg_idx, neg_max2

            N = output_ul_all.size(0)
            logits2_down = torch.zeros(pos2.size(0)).float().cuda()
            for i in range((N - 1) // b + 1):
                pseudo_label_idx = pseudo_label_all[i * b:(i + 1) * b]
                output_ul_idx = output_ul_all[i * b:(i + 1) * b]
                if i == 0:
                    logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(run2_0, pos2, output_feat2, output_ul_idx,
                                                                                  pseudo_label_idx, pseudo_label2_overlap)
                else:
                    logits2_neg_idx = torch.utils.checkpoint.checkpoint(run2, pos2, output_feat2, output_ul_idx,
                                                                        pseudo_label_idx, pseudo_label2_overlap, neg_max2)
                logits2_down += logits2_neg_idx

            logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)
            pos_mask_2 = ((pseudo_logits1_overlap > self.pos_thresh_value) & (
                    pseudo_logits2_overlap < pseudo_logits1_overlap)).float()
            loss2 = -torch.log(logits2 + eps)
            loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)

            loss_unsup = self.weight_unsup * (loss1 + loss2)
            mix_loss = mix_loss + loss_unsup
            mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

            if seg_debug['Source'] is not None and seg_debug:
                for j in range(batch_size):
                    rows, cols = 3, len(seg_debug['Source'])
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            if out.shape[1] == 3:
                                vis = torch.clamp(
                                    denorm(out, means, stds), 0, 1)
                                subplotimg(axs[k1][k2], vis[j], f'{n1} {n2}')
                            else:
                                if out.ndim == 3:
                                    args = dict(cmap='cityscapes')
                                else:
                                    args = dict(cmap='gray', vmin=0, vmax=1)
                                subplotimg(axs[k1][k2], out[j], f'{n1} {n2}',
                                           **args)
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
        self.local_iter += 1

        return log_vars
