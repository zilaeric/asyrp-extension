
# On the Reproduction and Extension of <br>"Diffusion Models already have a Semantic Latent Space"

[![arXiv](https://img.shields.io/badge/arXiv-2110.02711-red)](https://arxiv.org/abs/2210.10960) [![project_page](https://img.shields.io/badge/project_page-orange)](https://kwonminki.github.io/Asyrp/)


> **Diffusion Models already have a Semantic Latent Space**<br>
> [Mingi Kwon](https://drive.google.com/file/d/1d1TOCA20KmYnY8RvBvhFwku7QaaWIMZL/view?usp=share_link), [Jaeseok Jeong](https://drive.google.com/file/d/14uHCJLoR1AFydqV_neGjl1H2rjN4HBdv/view), [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor) <br>
> Arxiv preprint.
> 
>**Abstract**: <br>
Diffusion models achieve outstanding generative performance in various domains. Despite their great success, they lack semantic latent space which is essential for controlling the generative process. To address the problem, we propose asymmetric reverse process (Asyrp) which discovers the semantic latent space in frozen pretrained diffusion models. Our semantic latent space, named h-space, has nice properties for accommodating semantic image manipulation: homogeneity, linearity, robustness, and consistency across timesteps. In addition, we introduce a principled design of the generative process for versatile editing and quality boosting by quantifiable measures: editing strength of an interval and quality deficiency at a timestep. Our method is applicable to various architectures (DDPM++, iDDPM, and ADM) and datasets (CelebA-HQ, AFHQ-dog, LSUN-church, LSUN-bedroom, and METFACES).
 
# Overview
### J. R. Gerbscheid, A. Ivășchescu, L. P. J. Sträter, E. Zila

This repository is a reproduction & extension effort of the of [Diffusion Models already have a Semantic Latent Space](https://arxiv.org/abs/2210.10960). 

This is a landing page for the project with installation & training instructions. For the full report, including our results, additional ablations & analysis and contributions please checkout our [blogpost](blogpost.md)

## Requirements

To install requirements please create a new conda environment from the environment.yml file (example taken from [src/jobs/install_env.job](src/jobs/install_env.job):

```setup
conda env create -f environment.yml
```
You will also need to download the data, below is an exmaple of how to load the celeba dataset and pretrained weights for it (example taken from [src/jobs/download_data.sh](src/jobs/download_data.sh)). 
```
source activate asyrp

bash src/lib/utils/data_download.sh celeba_hq src/
rm a.zip

mkdir src/lib/asyrp/pretrained/
python src/lib/utils/download_weights.py
```

## Training

To train the models, refer to the ablation scripts in `src/lib/asyrp/ablations`.

for example: [src/lib/asyrp/scripts/ablations/simple_transformer/nheads/script_train_tranformer_pixar_pc_h8_d_2048.sh](src/lib/asyrp/scripts/ablations/simple_transformer/nheads/script_train_tranformer_pixar_pc_h8_d_2048.sh)


## Evaluation

To evaluate the models, use the same scripts as mentioned before but set the `do_test` argumet to 1. 

You can use the reproduction scripts in `src/lib/asyrp/reproductions`, such as [src/lib/asyrp/scripts/reproduction/script_inference_frida.sh](src/lib/asyrp/scripts/reproduction/script_inference_frida.sh)
Finally, to evaluate the additional metrics, FID and Directional CLip Loss refer to the evaluation notebook [demos/ablations/ablation_metrics.ipynb](demos/ablations/ablation_metrics.ipynb)
