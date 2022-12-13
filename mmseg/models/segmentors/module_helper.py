#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pdb

import torch
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


from torch.nn.functional import interpolate


class ModuleHelper(object):

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == 'torchbn':
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'torchsyncbn':
            return nn.Sequential(
                nn.SyncBatchNorm(num_features, **kwargs),
                nn.ReLU()
            )

        else:
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type='torch', ret_cls=False):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d

        elif bn_type == 'torchsyncbn':
            return nn.SyncBatchNorm

        elif bn_type == 'gn':
            return functools.partial(nn.GroupNorm, num_groups=32)

        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]


        else:

            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network='resnet101'):
        if pretrained is None:
            return model

        if all_match:

            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'resinit.{}'.format(k) in model_dict:
                    load_dict['resinit.{}'.format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:

            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)

            # settings for "wide_resnet38"  or network == "resnet152"
            if network == "wide_resnet":
                pretrained_dict = pretrained_dict['state_dict']

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                # pretrained_dict['conv1_full_res.weight'] = pretrained_dict['conv1.weight']
                # pretrained_dict['conv2_full_res.weight'] = pretrained_dict['conv2.weight']
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pvt':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                pretrained_dict['pos_embed1'] = \
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed2'] = \
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed3'] = \
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                pretrained_dict['pos_embed4'] = \
                    interpolate(pretrained_dict['pos_embed4'].unsqueeze(dim=0), size=[256, 512])[0]
                pretrained_dict['pos_embed7'] = \
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed6'] = \
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed5'] = \
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pcpvt' or network == 'svt':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}


            elif network == 'transunet_swin':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                for item in list(pretrained_dict.keys()):
                    if item.startswith('layers.0') and not item.startswith('layers.0.downsample'):
                        pretrained_dict['dec_layers.2' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.1') and not item.startswith('layers.1.downsample'):
                        pretrained_dict['dec_layers.1' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.2') and not item.startswith('layers.2.downsample'):
                        pretrained_dict['dec_layers.0' + item[15:]] = pretrained_dict[item]

                for item in list(pretrained_dict.keys()):
                    if 'relative_position_index' in item:
                        pretrained_dict[item] = \
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[256, 256])[0][0]
                    if 'relative_position_bias_table' in item:
                        pretrained_dict[item] = \
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[961, pretrained_dict[item].size(1)])[0][0]
                    if 'attn_mask' in item:
                        pretrained_dict[item] = \
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[pretrained_dict[item].size(0), 256, 256])[0][0]

            elif network == "hrnet" or network == "xception" or network == 'resnest':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}


            elif network == "dcnet" or network == "resnext":
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if 'resinit.{}'.format(k) in model_dict:
                        load_dict['resinit.{}'.format(k)] = v
                    else:
                        if k in model_dict:
                            load_dict[k] = v
                        else:
                            pass

            elif network == "wide_resnet":
                load_dict = {'.'.join(k.split('.')[1:]): v \
                             for k, v in pretrained_dict.items() \
                             if '.'.join(k.split('.')[1:]) in model_dict}
            else:
                load_dict = {'.'.join(k.split('.')[1:]): v \
                             for k, v in pretrained_dict.items() \
                             if '.'.join(k.split('.')[1:]) in model_dict}


        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.PyTorchCV', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            urlretrieve(url, cached_file)


        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
