# Robust Unsueprvised StyleGAN Image Restoration

This is a preliminary code release. Launch scripts & instructions are coming soon.

<!-- 
## Installation

First install the same environment as https://github.com/NVlabs/stylegan2-ada-pytorch.git.
For evaluation you will need to run `pip install torchmetetric git+https://github.com/jwblangley/pytorch-fid.git`.
 todo rglob 

Then, download the pretrained StyleGAN model:
```
(cd pretrained; wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl)
```

Place your images in `datasets`. A few sample images are already in `dataset/samples`.

## Restoring images

In `restoration_config.py`, specify your dataset location and image resolution. Then, run `python restore_images.py`. This will produce results in the `out` folder.

On datasets other than faces you may need to scale all learning rates up or down by a constant amount to compensate for the different scale of the latent space.

## Running the benchmark
Run `python eval_restoration.py out/$NAME` where `$NAME` is the folder name for your expriment, which can be specified in `restoration_config.py`. -->