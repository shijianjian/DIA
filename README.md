# Dissolving Is Amplifying: Towards Fine-Grained Anomaly Detection [ECCV2024]


<div align="center">
Jian Shi<sup>1</sup> · Pengyi Zhang<sup>2</sup> · Ni Zhang<sup>2</sup> · Hakim Ghazzai<sup>1</sup> ·  Peter Wonka<sup>1</sup>


<sup>1</sup>King Abdullah University of Science and Technology                                                                
<sup>2</sup>NEC Laboratories China
    
<a href="https://arxiv.org/abs/2302.14696"><img src='https://img.shields.io/badge/arXiv-Dissolving Is Amplifying-red' alt='Paper PDF'></a>
<a href='https://shijianjian.github.io/DIA/'><img src='https://img.shields.io/badge/Project_Page-Dissolving Is Amplifying-green' alt='Project Page'></a>
</div>



<p align="center">
    <img src=https://shijianjian.github.io/DIA/static/images/main-graph.png width="900"> 
</p>


DIA is a fine-grained anomaly detection framework for medical images. We describe two novel components.

- First, the dissolving transformations. Our main observation is that generative diffusion models are feature-aware and applying them to medical images in a certain manner can remove or diminish fine-grained discriminative features such as tumors or hemorrhaging. More visual demonstration about the dissolving effects are [here](https://shijianjian.github.io/DIA/).
- Second, an amplifying framework. It is based on contrastive learning to learn a semantically meaningful representation of medical images in a self-supervised manner.

The amplifying framework contrasts additional pairs of images with and without dissolving transformations applied and thereby boosts the learning of fine-grained feature representations. DIA significantly improves the medical anomaly detection performance with around 18.40\% AUC boost against the baseline method and achieves an overall SOTA against other benchmark methods.

---
**News**
- [July 2024] We are integrating to [Kornia](https://github.com/kornia/kornia)! You may access StableDiffusion-based dissolving transformations with `kornia.filters.StableDiffusionDissolving` or use it as an augmentation `kornia.augmentation.RandomDissolving`. Kornia welcomes contributors for their AI-based light-weight operations! 
- [July 2024] Our paper is accepted to ECCV2024!
---

## 1. Requirements
### Environments
```
$ pip install -r requirements.txt
```

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

- Other transformation types are available to use, e.g. `diffusion_rotation`, `diffusion_cutperm`, `blurgaussian_rotation`, `blurgaussian_cutperm`, `blurmedian_rotation`, `blurmedian_cutperm`. Note that cutperm may perform better if your dataset is more *rotation-invariant*, while the rotation may perform better if the dataset is more "permutation-invariant".
- To test different resolutions, do remember to re-train the diffusion models for the corresponding resolution.
- For low resolution datasets, e.g. CIFAR10, use `--model resnet18` instead of `--model resnet18_imagenet`.

## 3. Evaluation

During the evaluation, we use only `rotation` shift transformation for evaluation.

```bash
python eval.py --mode ood_pre --dataset <DATASET> --model resnet18_imagenet --ood_score CSI --shift_trans_type rotation --print_score --diffusion_upper_offset 0. --diffusion_offset 0. --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 0 --load_path <MODEL_PATH>
```

## 4. Results

In general, our method achieves SOTA in terms of fine-grained anomaly detection tasks. We recommend using 32x32 diffusion_rotation for most tasks, as the following results:

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

Higher feature dissolving resolution will dramatically increase the processing time, while hardly bringing up the detection performances.


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
  Author = {Jian Shi and Pengyi Zhang and Ni Zhang and Hakim Ghazzai and Peter Wonka},
  Title = {Dissolving Is Amplifying: Towards Fine-Grained Anomaly Detection},
  Year = {2023},
  Eprint = {arXiv:2302.14696},
}
```

## Acknowledgement
Our method is based on [CSI](https://github.com/alinlab/CSI).
