# Checkpoints

Place the following pretrained backbone checkpoints in this directory before running feature extraction or training:

- `Cnn14_16k_mAP=0.438.pth` — PANNs (Pretrained Audio Neural Networks) checkpoint, used by the Audio Semantic Encoder. Download from the [PANNs release](https://github.com/qiuqiangkong/audioset_tagging_cnn).
- `resnet-34-kinetics.pth` — 3D ResNet-34 pretrained on Kinetics-400, used as the TVSum visual backbone. Download from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).

Trained model checkpoints (`best_model_*.pt`) produced by `main.py` are written under `$OUTPUT_ROOT` (see `model/configs.py`), not here.
