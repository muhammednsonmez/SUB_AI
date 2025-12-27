# SUBAI Auto Subtitler

A Python tool that automatically generates and burns subtitles into videos using the Faster-Whisper model. It detects speech, transcribes it, and hardcodes the subtitles directly onto the video file.

## Features

- **High Performance:** Ultra-fast transcription using the Faster-Whisper library.
- **Language Support:** Automatic language detection with manual override options.
- **Hardware Acceleration:** Supports both CPU and GPU (CUDA) with automatic fallback.
- **Smart Timing:** Uses word-level timestamps to hide subtitles during silence.
- **Text Cleaning:** Automatically removes music notes, emojis, and sound effect descriptions from the subtitles.

## Installation

1. Clone or download this repository.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt

## Additional Requirements
FFmpeg: FFmpeg must be installed on your system or the binaries must be present in the ffmpeg/bin folder within the project directory.

CUDA (Optional): To use GPU acceleration, NVIDIA CUDA Toolkit 12.x and zlibwapi.dll are required.

[Download FFmpeg](https://www.ffmpeg.org/download.html)
[Download ZLIBWAPI.dll](https://www.winimage.com/zLibDll/index.html)
[Download CUDA](https://developer.nvidia.com/cuda-12-6-0-download-archive)

## Usage

Place your video files (e.g., `.mp4`, `.mov`, `.avi`) into the **`inputs`** folder.

Run the main script to start the interactive menu:

    ```bash
    python main.py
    ```

Follow the on-screen instructions to select your video, choose the model size, and configure language settings.
Once completed, find the subtitled video and .srt file in the outputs folder.