
## Testing

We also provide pretrained models below for PiPa based on DAFormer.

### 

| model name                       | mIoU  | checkpoint file download                    |
| :------------------------------- | :---: | :------------------------------------- |
| pipa_gta_to_cs.pth  | 71.7  | [Google Drive](https://drive.google.com/file/d/1qDAiS1gzhkFgoPwrLcJXlXgjr8OysH0h/view?usp=share_link)|
| pipa_syn_to_cs.pth  | 63.4  | [Google Drive](https://drive.google.com/file/d/1iQWBrrvwCFaPdg6a9bnlthmGYPyTy0y4/view?usp=share_link)|


```shell
python -m tools.test path/to/config_file path/to/checkpoint_file --format-only --eval-option 
```


## Training

```shell
python run_experiments.py --config configs/pipa/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
```

The logs and checkpoints are stored in `work_dirs/`.

