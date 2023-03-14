# Dissolving Is Amplifying: Towards Fine-Grained Anomaly Detection

Official PyTorch implementation of
["**Dissolving Is Amplifying: Towards Fine-Grained Anomaly Detection**"](
https://arxiv.org/pdf/2302.14696)

<p align="center">
    <img src=figures/dissolving.PNG width="900"> 
    <em>Demonstration of the diffusion feature dissolving outcome. Input images are converged to domain-specific generalized patterns by dissolving instance features whilst going through the diffusion feature dissolving process.</em>
</p>


## Abstract

Medical anomalous data normally contains fine-grained
instance-wise additive feature patterns (e.g. tumor, hemorrhage), that are oftenly critical but insignificant. Interestingly, apart from the remarkable image generation abilities of diffusion models, we observed that diffusion models can well-dissolve image details for a given image, resulting in generalized feature representations. We hereby propose DIA, dissolving is amplifying, that amplifies finegrained image features by contrasting an image against its feature dissolved counterpart. In particular, we show that diffusion models can serve as semantic preserving feature dissolvers that help learning fine-grained anomalous patterns for anomaly detection tasks, especially for medical domains with fine-grained feature differences. As a result, our method yields a novel fine-grained anomaly detection method, aims at amplifying instance-level feature patterns, that significantly improves medical anomaly detection accuracy in a large margin without any prior knowledge of explicit fine-grained anomalous feature patterns.

<p align="center">
    <img src=figures/mainfig.PNG width="900"> 
</p>

## 1. Requirements
### Environments
The experiment environment setting is:
- apex==0.9.10.dev0
- denoising_diffusion_pytorch==0.27.12
- diffdist==0.1
- kornia==0.6.8
- matplotlib==3.6.0
- medmnist==2.1.0
- numpy==1.24.2
- opencv_python==4.6.0.66
- Pillow==9.4.0
- scikit_learn==1.2.2
- tensorboardX==2.6
- thop==0.1.1.post2209072238
- torch==1.11.0
- torchlars==0.1.2
- torchvision==0.12.0

### Datasets 
We majorly use the following datasets to benchmark our method:
- [APTOS2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data),
- [SARS-COV-2](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)
- [Retina OCT](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [Kvasir](https://datasets.simula.no/kvasir/)


## 2. Training

### Pretraining of diffusion models
Modify the data paths in `diffusion_models/train_eval_ddpm.py`, then run:
```
python train_eval_ddpm.py --mode train --dataset <DATASET>
```
In our experiments, the diffusion models are usable within 5 epochs.

### Training DIA

To train unlabeled one-class & multi-class models in the paper, run this command:

```bash
python train.py --data_root ../data --dataset <DATASET> --model resnet18_imagenet --mode simclr_DIA --shift_trans_type diffusion_rotation --diff_resolution 32 --batch_size 32 --one_class_idx 0 --save_step 1  --diffusion_model_path <DIFFUSION_MODEL_PATH>
```

- Other transformation types are avaliable to use, e.g. `diffusion_rotation`, `diffusion_cutperm`, `blurgaussian_rotation`, `blurgaussian_cutperm`, `blurmedian_rotation`, `blurmedian_cutperm`.
- To test different resolutions, do remember to re-train the diffusion models for the corresponding resotution.
- For low resolution datasets, e.g. CIFAR10, use `resnet18` instead of `resnet18_imagenet`.

## 3. Evaluation

During the evaluation, we use only `rotation` shift transformation for evaluation.

```bash
python eval.py --mode ood_pre --dataset <DATASET> --model resnet18_imagenet --ood_score CSI --shift_trans_type rotation --print_score --diffusion_upper_offset 0. --diffusion_offset 0. --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 0 --load_path <MODEL_PATH>
```

## 4. Results

In general, our method achieves SOTA in terms of the fine-grained anomaly detection tasks. We recommend using 32x32 diffusion_rotation for most tasks, as the following results:

### Different transformations

In general, `diffusion_rotation` performs best.


| Dataset       | transform    | Gaussian     |   Median  |  Diffusion |
| --------------|------------- | ------------ | --------- | ---------- |
| SARS-COV-2    | Perm         |    0.788     |   0.826   |   0.841    |
|               | Rotate       |    0.847     |   0.837   |   0.851    |
| Kvasir        | Perm         |    0.712     |   0.663   |   0.840    |
|               | Rotate       |    0.739     |   0.687   |   0.860    |
| Retinal OCT   | Perm         |    0.754     |   0.747   |   0.890    |
|               | Rotate       |    0.895     |   0.876   |   0.919    |
| APTOS 2019    | Perm         |    0.942     |   0.929   |   0.926    |
|               | Rotate       |    0.922     |   0.918   |   0.934    |


### Resolutions

Higher feature dissolving resolution will dramtically increase the processing time, while hardly bring up the detection performances.


| Dataset       | resolution   |   MACs (G)   |     AUC      |
| --------------|------------- |------------- | ------------ |
| SARS-COV-2    | 32           |  2.33        |    0.851     |
|               | 64           |  3.84        |    0.803     |
|               | 128          |  9.90        |    0.807     |
| Kvasir        | 32           |  2.33        |    0.860     |
|               | 64           |  3.84        |    0.721     |
|               | 128          |  9.90        |    0.730     |
| Retinal OCT   | 32           |  2.33        |    0.919     |
|               | 64           |  3.84        |    0.922     |
|               | 128          |  9.90        |    0.930     |
| APTOS 2019    | 32           |  2.33        |    0.934     |
|               | 64           |  3.84        |    0.937     |
|               | 128          |  9.90        |    0.905     |


## Citation
```
@misc{2302.14696,
  Author = {Jian Shi and Pengyi Zhang and Ni Zhang and Hakim Ghazzai and Yehia Massoud},
  Title = {Dissolving Is Amplifying: Towards Fine-Grained Anomaly Detection},
  Year = {2023},
  Eprint = {arXiv:2302.14696},
}
```

## Acknowledgement
Our method is heavily modified from [CSI](https://github.com/alinlab/CSI).
