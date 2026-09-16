<h2 align="center"> <a href="https://arxiv.org/abs/2602.03891">Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection</a></h2>

<h4 align="center"> <a href="https://www.linkedin.com/in/seohyun-joo-36a56929b">Seohyun Joo</a>, <a href="https://yoori000.github.io">Yoori Oh</a> </h4>

<h4 align="center"> [<a href="https://seohyj.github.io/soundhd.github.io/">🌐 Project Page</a>] [<a href="https://arxiv.org/pdf/2602.03891">📖 Paper</a>] [<a href="https://arxiv.org/abs/2602.03891">arXiv</a>] </h4>

<p align="center">Official implementation of <b>"Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection"</b> (ICASSP 2026)</p>

<div align="center"><img width="95%" alt="DAViHD framework" src="figs/architecture.png?raw=true"/></div>

## 📊 Overview

- We propose **DAViHD**, an audio-visual video highlight detection model built around a **Dual-Pathway Audio Encoder**: an *Audio Semantic Encoder* (PANNs embeddings — *what* is heard) alongside an *Audio Dynamics Encoder* (a frequency-dynamic convolution over the log-mel spectrogram — *how* the sound changes over time).
- The two audio streams are fused and combined with the visual stream through bidirectional cross-modal attention to produce a per-frame highlight score.
- We show that spectro-temporal audio dynamics — largely ignored by prior audio-visual highlight detection work — are highly effective for identifying highlights, and that DAViHD outperforms video-only and prior audio-visual baselines on both TVSum and Mr.HiSum.

## 🛠️ Requirements

- Python 3.9, PyTorch 2.1.0
- FFmpeg 6 from conda-forge — torchaudio decodes audio directly from the video files, and torchaudio 2.1.0 does not support FFmpeg 7's ABI

```bash
conda create -n avhd python=3.9
conda activate avhd
conda install -c conda-forge "ffmpeg=6"
pip install -r requirements.txt
```

A Dockerfile with the same stack is provided:

```bash
docker build -t avhd .
docker run --gpus all -v /path/to/data:/AV-HD/data -v /path/to/checkpoints:/AV-HD/checkpoints avhd --dataset mrhisum --mode train
```

Paths are set through environment variables: `DATA_ROOT` (extracted features, default `./data`), `OUTPUT_ROOT` (runs, checkpoints, logs, default `./outputs`) and `CHECKPOINT_ROOT` (pretrained backbones, default `./checkpoints`).

## 📦 Dataset

### Pretrained backbones

Download these into `$CHECKPOINT_ROOT`:

| File | Used by | Source |
|---|---|---|
| `Cnn14_16k_mAP=0.438.pth` | Audio Semantic Encoder | [PANNs release](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| `resnet-34-kinetics.pth` | TVSum visual backbone | [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) |

### TVSum

Download the dataset from [people.csail.mit.edu/yalesong/tvsum](https://people.csail.mit.edu/yalesong/tvsum) and place it so that `$DATA_ROOT/tvsum/` holds `video/`, `matlab/ydata-tvsum50.mat` and `data/ydata-tvsum50-anno.tsv`. Then:

```bash
python dataset/preprocess/tvsum_matlab.py                             # metadata
python dataset/preprocess/tvsum_gtscore.py                            # ground-truth scores
python dataset/preprocess/tvsum_extract_visual_features.py            # 3D-ResNet-34 visual features
python dataset/preprocess/extract_audio_features.py --dataset tvsum   # PANNs audio embeddings
python dataset/preprocess/extract_melspec.py --dataset tvsum          # log-mel spectrograms
```

### Mr.HiSum

Download `metadata.csv` and `mr_hisum.h5` from the official [MR.HiSum](https://github.com/MRHiSum/MR.HiSum) repository, place `metadata.csv` at `$DATA_ROOT/mrhisum/metadata.csv`, and fetch the videos with `dataset/crawler.py`. Then:

```bash
python dataset/preprocess/pca_matrix_to_npy.py                        # once
python dataset/preprocess/mrhisum_extract_visual_features.py          # InceptionV3 + PCA
python dataset/preprocess/extract_audio_features.py --dataset mrhisum
python dataset/preprocess/extract_melspec.py --dataset mrhisum
```

`mrhisum_npy_to_h5.py` / `mrhisum_h5_to_npy_*.py` convert between the per-video `.npy` and packed HDF5 layouts (`--use_hdf5`). `dataset/mrhisum_split.json` holds the train/val/test split used in the paper.

## 🚀 Training

```bash
# Mr.HiSum
python main.py --dataset mrhisum --mode train \
    --epochs 200 --batch_size 16 --lr 1e-5 --l2_reg 1e-4 --grad_clip 0.5 --use_hdf5

# TVSum, 5-fold cross-validation over 5 seeds
python main.py --dataset tvsum --mode train \
    --epochs 400 --batch_size 8 --lr 5e-6 --l2_reg 1e-4 --grad_clip 0.5 --repeat
```

Training writes `best_model.pt`, the epoch with the best validation F1, which `main.py` reloads to report the test scores. `--repeat` additionally writes `final_summary.txt` and a per-seed `run_summary.txt`.

## 💾 Model Checkpoints

Trained TVSum checkpoints (`tvsum_fold1_seed42.pt` … `tvsum_fold5_seed42.pt`, seed 42) are available on [Google Drive](https://drive.google.com/drive/folders/1V5XdNwYA7hQPgnAPueQu9y90xlQwAPWp?usp=sharing).

TVSum is evaluated with 5-fold cross-validation, so each checkpoint was trained on 40 videos and holds out a different 10. `fold_splits.txt` in the same folder lists which 10 belong to each fold — evaluate a checkpoint only on those, since the rest were its training data.

## 🔍 Inference

```bash
python inference/predict.py --ckpt_path <path/to/best_model.pt> \
    --audio_feat_dir <...> --visual_feat_dir <...> --melspec_dir <...> --output_dir <...> --dataset mrhisum
```

## 📁 Repository layout

```text
main.py                    # Training/eval entry point
model/                     # Config, training loop, evaluation metrics
dataset/loaders/           # PyTorch Dataset/DataLoader implementations
dataset/preprocess/        # Feature-extraction scripts
networks/                  # DAViHD architecture and pretrained backbones
inference/                 # Standalone prediction script
checkpoints/               # Pretrained backbone checkpoints
```

## 🙏 Acknowledgement

This implementation builds upon [FDY-SED](https://github.com/frednam93/FDY-SED), [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) and [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch). We appreciate the open source of the projects; see [`NOTICE.md`](NOTICE.md) for their licenses.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{joo2026sounding,
  title={Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection},
  author={Joo, Seohyun and Oh, Yoori},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={13027--13031},
  year={2026},
  organization={IEEE}
}
```
