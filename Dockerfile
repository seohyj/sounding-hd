FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# torchaudio decodes audio straight out of the video files through its ffmpeg backend, which
# needs FFmpeg 6 — torchaudio 2.1.0 does not support FFmpeg 7's ABI, and the distro package is
# too old. This mirrors the conda setup in the README.
RUN conda install -y -c conda-forge "ffmpeg=6" && conda clean -afy

WORKDIR /AV-HD

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
