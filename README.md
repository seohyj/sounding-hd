<h2 align="center"> <a href="https://arxiv.org/abs/2602.03891">Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection</a></h2>

<h4 align="center"> <a href="https://www.linkedin.com/in/seohyun-joo-36a56929b">Seohyun Joo</a>, <a href="https://yoori000.github.io">Yoori Oh</a> </h4>

<h4 align="center"> [<a href="https://seohyj.github.io/soundhd.github.io/">🌐 Project Page</a>] [<a href="https://arxiv.org/pdf/2602.03891">📖 Paper</a>] [<a href="https://arxiv.org/abs/2602.03891">arXiv</a>] </h4>

<p align="center">Official implementation of <b>"Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection"</b> (ICASSP 2026)</p>

## 📊 Overview

- We propose **DAViHD**, an audio-visual video highlight detection model built around a **Dual-Pathway Audio Encoder**: an _Audio Semantic Encoder_ (PANNs embeddings — _what_ is heard) alongside an _Audio Dynamics Encoder_ (a frequency-dynamic convolution over the log-mel spectrogram — _how_ the sound changes over time).
- The two audio streams are fused and combined with the visual stream through bidirectional cross-modal attention to produce a per-frame highlight score.
- We show that spectro-temporal audio dynamics — largely ignored by prior audio-visual highlight detection work — are highly effective for identifying highlights, and that DAViHD outperforms video-only and prior audio-visual baselines on both TVSum and Mr.HiSum.
