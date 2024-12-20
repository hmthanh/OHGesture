
![gesture_generation.png](./gesture_generation.png)

# OHGesture - A conversational gesture synthesis system based on emotions and semantics

[![Hugging Face Pretrain](https://img.shields.io/badge/Pretrain-Hugging%20Face-blue)](https://huggingface.co/openhuman/openhuman)
[![Homepage](https://img.shields.io/badge/Homepage-green)](https://deepgesture.github.io)

![OHGesture](./OHGesture.png)

## Quick Start

```bash
pip install -r requirements.txt
```

* Download pretrained model [OneDrive](https://1drv.ms/f/s!AvSTDY2o11xHgalWGd7PGtdj5yOiRA?e=xek1oW) and put in  `./main/`


## Preprocessing data

### zeggs_data_to_h5

```bash
cd ./ZeroEGGSProcessing
python zeggs_data_to_h5.py
```

### Convert word to vector

```bash
cd ./ZeroEGGSProcessing
python word2vec.py --src=./data/train  --dest=./processed/train/embedding --word2vec_model=./fasttext/crawl-300d-2M.vec
```

### Convert data to h5dataset

```bash
cd ./ZeroEGGSProcessing
python data_to_h5dataset.py --config=../main/configs/OHGesture.yml
```

## Training

```bash
cd ./main
python ./end2end.py --config=./configs/DiffuseStyleGesture.yml -gpu=cuda:0
```

## Visualization

Clone [[https://github.com/DeepGesture/deepgesture-unity]](https://github.com/DeepGesture/deepgesture-unity)

```shell
git clone https://github.com/DeepGesture/deepgesture-unity
```

Open with Unity 6 and import BVH to visualize result.

## Reference

This source code is based on the following repositories:
[[YoungSeng/DiffuseStyleGesture]](https://github.com/YoungSeng/DiffuseStyleGesture)

## Citation

```bibtex

```
