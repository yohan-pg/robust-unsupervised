# Robust Unsupervised StyleGAN Image Restoration
### [[Arxiv]](https://arxiv.org/abs/2302.06733) [[Website]](https://lvsn.github.io/RobustUnsupervised/)

Code for the paper `Robust Unsupervised StyleGAN Image Restoration` presented at CVPR 2023. 

## Installation

1) First install the same environment as https://github.com/NVlabs/stylegan2-ada-pytorch.git. It is not essential for the custom cuda kernels to compile correctly, they just make things run ~30% faster.

2) Run `pip install tyro`. For running the evaluation you will also need to `pip install torchmetrics git+https://github.com/jwblangley/pytorch-fid.git`.

2) Download the pretrained StyleGAN model:
```bash
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -O pretrained/ffhq.pkl
```
## Restoring images

To run the tasks presented in the paper, use:

```bash 
python run.py --dataset_path datasets/samples
```

Some sample images have already been provided in `datasets/samples`.

## Other datasets
First, download a pretrained StyleGAN2 generator for your dataset (.pkl), and pass it's path to the `--pkl_path` option.
If the resolution of your data is different from 1024 you also need to set it using the `--resolution` option.
This resolution does not need to match the pretrained generator's resolution; for best results pick a high resolution generator even if your images are smaller. 

Finally, on datasets other than faces you may need to scale all learning rates up or down by a constant amount to compensate for the different scale of the latent space. For this you can use the CLI option `--global_lr_scale`.

## Restoring your own degradations
Use the option `--tasks custom`, then find the following code in `run.py` and update it with your degradation function:

```python
class YourDegradation:
    def degrade_ground_truth(self, x):
        """
        The true degradation you are attempting to invert.
        This assumes you are testing against clean ground truth images.
        """
        raise NotImplementedError
    
    def degrade_prediction(self, x):
        """
        Differentiable approximation to the degradation in question. 
        Can be identical to the true degradation if it is invertible.
        """
        raise NotImplementedError
```
If you do not have access to ground truth images, you can open degraded images directly and make `degrade_ground_truth` an indentity function.

## Evaluation
To run the full evaluation, use: 
```
python -m benchmark.eval
```
Due to random variability the numbers may not match the paper exactly, but you should expect scores to be equal or better on average. For instance:
```
XL Upsampling: 21.5 (this repo) vs. 21.3 (paper)
XL Denoising: 17.8 (this repo) vs. 17.9 (paper)
XL Deartifacting: 16.7 (this repo) vs. 18.7 (paper)
XL Inpainting: 14.0 (this repo) vs 15.0 (paper)
```
