# Robust Unsueprvised StyleGAN Image Restoration
<!-- todo arxiv and website links -->

Code for the paper ......

## Installation

1) First install the same environment as https://github.com/NVlabs/stylegan2-ada-pytorch.git. It is not essential for the custom cuda kernels to compile correctly, they just make things run ~30% faster.

2) Run `pip install tyro`. For running the evaluation you will also need to `pip install torchmetrics git+https://github.com/jwblangley/pytorch-fid.git`.

2) Download the pretrained StyleGAN model:
```bash
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -O pretrained/ffhq.pkl
```
## Restoring images

To run the restoration on degraded images, use

```bash 
python restore.py --dataset_path datasets/samples/$TASK --task $TASK
```
Where `$TASK` can be `deartifact`, `denoise`, `upsample`, or `inpaint`. 

Results will be saved in the `out` folder.

## Other degradations

## Other datasets
First, download a pretrained StyleGAN2 generator for your dataset (.pkl), and pass it's path to the `--pkl_path` option.
If the resolution is different from 1024 you also need to set it using the `--resolution` option.

Finally, on datasets other than faces you may need to scale all learning rates up or down by a constant amount to compensate for the different scale of the latent space. For this you can use the CLI option `--global_lr_scale`.

## Using your own degradation functions
The codebase was designed to run ...........
If you want to run on your own 

## 
Coming soon.
<!-- For evaluation purposes, the paper used synthetic degradations which were applied on-the-fly to clean images. For this you can use:
`python restore_synthetic.py --dataset_path datasets/samples/raw`
where `datasets/samples/raw` contains clean images.  -->
<!-- Run `python eval_restoration.py out/$NAME` where `$NAME` is the folder name for your expriment, which can be specified in `restoration_config.py`. --> 



