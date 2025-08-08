<div align="center">

<h1>MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization</h1>

[![arXiv](https://img.shields.io/badge/ArXiv-2507.07997-%23840707.svg)](https://arxiv.org/abs/2507.07997) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mgvq-could-vq-vae-beat-vae-a-generalizable/image-reconstruction-on-imagenet)](https://paperswithcode.com/sota/image-reconstruction-on-imagenet?p=mgvq-could-vq-vae-beat-vae-a-generalizable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mgvq-could-vq-vae-beat-vae-a-generalizable/image-reconstruction-on-ultra-high-resolution)](https://paperswithcode.com/sota/image-reconstruction-on-ultra-high-resolution?p=mgvq-could-vq-vae-beat-vae-a-generalizable)

[Mingkai Jia](https://scholar.google.com/citations?user=fcpTdvcAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Wei Yin](https://yvanyin.net/)<sup>2*§</sup>, [Xiaotao Hu](https://huxiaotaostasy.github.io/)<sup>1,2</sup>, [Jiaxin Guo](https://wrld.github.io/)<sup>3</sup>, [Xiaoyang Guo](https://xy-guo.github.io/)<sup>2</sup><br>
[Qian Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=pCY-bikAAAAJ)<sup>2</sup>, [Xiao-Xiao Long](https://www.xxlong.site/)<sup>4</sup>, [Ping Tan](https://scholar.google.com/citations?user=XhyKVFMAAAAJ&hl=en)<sup>1</sup><br>

[HKUST](https://hkust.edu.hk/)<sup>1</sup>, [Horizon Robotics](https://en.horizon.auto/)<sup>2</sup>, [CUHK](https://cuhk.edu.hk/)<sup>3</sup>, [NJU](https://www.nju.edu.cn/)<sup>4</sup><br>
<sup>*</sup> Corresponding Author, <sup>§</sup> Project Leader
<br><br><image src="./assets/teaser.png"/>
</div>


## 🚀News

- ```[July 2025]``` Released [paper](https://arxiv.org/abs/2507.07997).

## 🔨TO DO LIST
- [ ] Training code.
- [ ] More demos.
- [x] Models & Evaluation code.
- [x] Huggingface models.
- [x] Release zero-shot reconstruction benchmarks.

## 🙈 Model Zoo
| Model | Downsample | Groups | Codebook Size | Training Data | Link |
|---|---|---|---|---|---|
|mgvq-f8c32-g4|8|4|32768|imagenet| [link](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f8c32_g4.pt) |
|mgvq-f8c32-g8|8|8|16384|imagenet| [link](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f8c32_g8.pt) |
|mgvq-f16c32-g4|16|4|32768|imagenet| [link](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f16c32_g4.pt) |
|mgvq-f16c32-g8|16|8|16384|imagenet| [link](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f16c32_g8.pt) |
|mgvq-f16c32-g4-mix|16|4|32768|mix| [link](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f16c32_g4_mix.pt) |
|mgvq-f32c32-g8-mix|32|8|16384|mix| [link](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f32c32_g8_mix.pt) |

## 🔑 Quick Start
<a id="quick start"></a>

### Installation

```bash
git clone https://github.com/MKJia/MGVQ.git
cd MGVQ
pip3 install requirements.txt
```

### Download models
Download the pretrained models from our [model zoo](https://huggingface.co/mkjia/MGVQ/tree/main) to your `/path/to/your/ckpt`.

### Data Preparation
Try our UHDBench dataset on [huggingface](https://huggingface.co/datasets/mkjia/UHDBench/tree/main) and download to your `/path/to/your/dataset`.

### Evaluation on Reconstruction
Remember to change the paths of `ckpt` and `dataset_root`, and make sure you are evaluating the expected `model` on `dataset`.
```bash
cd evaluation
python3 eval_recon.sh
```

### Generation Demo&Evaluation
You can download the pretrained GPT model for generation on [huggingface](https://huggingface.co/datasets/mkjia/MGVQ/blob/main/MGVQ_GPT_XXL.pt), and test it with our `mgvq-f16c32-g4` [tokenizer model](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f16c32_g4.pt) for demo image sampling. Remember to change the paths of `gpt_ckpt` and `vq_ckpt`. 
```
cd evaluation
python3 demo_gen.sh
```
We also provide our .npz file on [huggingface](https://huggingface.co/datasets/mkjia/MGVQ/blob/main/GPT_XXL_300ep_topk_12.npz) sampled by `sample_c2i_ddp.py` for evaluation.
```
cd evaluation
python3 evaluator.py /path/to/your/VIRTUAL_imagenet256_labeled.npz /path/to/your/GPT_XXL_300ep_topk_12.npz
```


## 🗄️Demos
- 🔥 Qualitative reconstruction images with $16$ x downsampling on $2560$ x $1440$ UHDBench dataset. 

<image src="./assets/qual_recon.png"/>

- 🔥 Qualitative class-to-image generation of Imagenet. The classes are dog(Golden Retriever and Husky), cliff, and bald eagle.

<image src="./assets/qual_gen.png"/>

## 📌 Citation

If the paper and code from `MGVQ` help your research, we kindly ask you to give a citation to our paper ❤️. Additionally, if you appreciate our work and find this repository useful, giving it a star ⭐️ would be a wonderful way to support our work. Thank you very much.

```bibtex
@article{jia2025mgvq,
  title={MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization},
  author={Jia, Mingkai and Yin, Wei and Hu, Xiaotao and Guo, Jiaxin and Guo, Xiaoyang and Zhang, Qian and Long, Xiao-Xiao and Tan, Ping},
  journal={arXiv preprint arXiv:2507.07997},
  year={2025}
}
```

## License

This repository is under the MIT License. For more license questions, please contact Mingkai Jia (mjiaab@connect.ust.hk) and Wei Yin (yvanwy@outlook.com).

