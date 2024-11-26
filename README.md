# OHGesture - Gesture Generation

## Quick Start

```bash
pip install -r requirements.txt
```

* Download pretrained model [OneDrive](https://1drv.ms/f/s!AvSTDY2o11xHgalWGd7PGtdj5yOiRA?e=xek1oW) and put in  `./mydiffusion_zeggs/`

```bash
cd ./diffuse_style_gesture
python ./sample.py --config=./configs/DiffuseStyleGesture.yml -gpu=cuda:0 --model_path="./model.pt" --speech_path "./021_Happy_4_x_1_0.wav"
```

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
python data_to_h5dataset.py --config=../mydiffusion_zeggs/configs/OHGesture.yml
```

## Training

```bash
cd ./diffuse_style_gesture
python ./end2end.py --config=./configs/DiffuseStyleGesture.yml -gpu=cuda:0
```

## Visualization

```shell
cd ./ubisoft-laforge-ZeroEGGS/ZEGGS/bvh2fbx
./bvh2fbx.bat
```

## Reference

This source code is based on the following repositories:
[[YoungSeng/DiffuseStyleGesture]](https://github.com/YoungSeng/DiffuseStyleGesture)

## Citation

```bibtex

```
