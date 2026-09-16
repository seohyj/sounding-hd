# Third-Party Notices

This repository's original code (the DAViHD model, training/eval pipeline, and dataset preprocessing scripts) is released under the MIT license (see `LICENSE`). It also includes code adapted or ported from the following third-party projects, which retain their own license terms — all of them permissive.

| Component | Source | License |
|---|---|---|
| `networks/tapconv/tfd_conv.py` (Frequency-Dynamic Convolution, used by the Audio Dynamics Encoder) | [frednam93/FDY-SED](https://github.com/frednam93/FDY-SED) | MIT |
| `networks/backbones/PANN.py`, `pann_utils.py` (audio backbone) | [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) — PANNs: Large-Scale Pretrained Audio Neural Networks | MIT |
| `networks/backbones/resnet.py` (TVSum visual backbone) | [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) | MIT |
| `dataset/preprocess/matrix_data.proto`, `matrix_data_pb2.py`, `inception3_projection_matrix_data.pb` (Mr.HiSum PCA matrices) | [google/youtube-8m](https://github.com/google/youtube-8m) | Apache-2.0 |
