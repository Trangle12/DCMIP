# [NOT Official] Discrepant and Multi-instance Proxies for Unsupervised Person Re-identification (ICCV2023)

[[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zou_Discrepant_and_Multi-Instance_Proxies_for_Unsupervised_Person_Re-Identification_ICCV_2023_paper.pdf)

## Upload History

* 2025/08/14: Unsupervised Code.

## Installation

Install `conda` before installing any requirements.

```bash
conda create -n dcmip python=3.9
conda activate dcmip
pip install -r requirements.txt
```
## 1. Introduction
Most recent unsupervised person re-identification methods maintain a cluster uni-proxy for contrastive learning.
However, due to the intra-class variance and inter-class
similarity, the cluster uni-proxy is prone to be biased and
confused with similar classes, resulting in the learned features lacking intra-class compactness and inter-class separation in the embedding space. To completely and accurately represent the information contained in a cluster
and learn discriminative features, we propose to maintain
discrepant cluster proxies and multi-instance proxies for a
cluster. Each cluster proxy focuses on representing a part
of the information, and several discrepant proxies collaborate to represent the entire cluster completely. As a complement to the overall representation, multi-instance proxies are used to accurately represent the fine-grained information contained in the instances of the cluster. Based on
the proposed discrepant cluster proxies, we construct cluster contrastive loss to use the proxies as hard positive samples to pull instances of a cluster closer and reduce intraclass variance. Meanwhile, instance contrastive loss is constructed by global hard negative sample mining in multiinstance proxies to push away the truly indistinguishable
classes and decrease inter-class similarity. Extensive experiments on Market-1501 and MSMT17 demonstrate that the
proposed method outperforms state-of-the-art approaches.

![image](./imgs/DCMIP.png)

## Datasets

Make a new folder named `data` under the root directory. Download the datasets and unzip them into `data` folder.
* [Market1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
* [MSMT17](https://arxiv.org/abs/1711.08565)

## Training

For example, training the full model on Market1501 with GPU 0 and saving the log file and checkpoints to `logs/market-pclclip`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -b 256 -d market1501 --iters 200 --eps 0.45 --logs-dir ./log/market1501

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -b 256 -d msmt17 --iters 200 --eps 0.7 --logs-dir ./log/msmt
```

## Results

The results are on Market1501 (M) and MSMT17 (MS).

| Methods | M | Link | MS | Link |
| --- | -- | -- | -- | - |
| CC + DCMIP | 86.7 (94.7) | - | 40.9 (69.3) | - |
| CC + DCMIP (Reproduce) |  () | - |  () | - |

## Note
The model training requires 4 GPUs.

The code is implemented based on following works.

1. [PCLHD](https://github.com/RikoLi/PCL-CLIP)
2. [NCPLR](https://github.com/haichuntai/NCPLR-ReID)
3. [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid)



