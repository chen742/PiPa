import itertools
import logging
import math


def get_model_base(architecture, backbone):
    architecture = architecture.replace('sfa_', '')
    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_half_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            cfg['_base_'].append(
                f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')
        if 'dacs' in uda and plcrop:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)

        # Setup optimizer and schedule
        if 'dacs' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU')

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
        if 'dacs' in uda and plcrop:
            uda_mod += '_cpl'
        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture_mod}_' \
                      f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn')
        return cfg

