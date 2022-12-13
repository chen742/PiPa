# DAFormer w/o DSC in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=False,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
