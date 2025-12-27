import os
import sys
import logging
import argparse
import subprocess
import re
import time
from pathlib import Path
from datetime import timedelta

# External Libraries
from faster_whisper import WhisperModel
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize Colors
init(autoreset=True)

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

# Directory Setup
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm'}

class VideoAutoSubtitler:
    def __init__(self, model_size="medium", device="auto", compute_type="int8"):
        self.model_size = model_size
        print(f"{Fore.CYAN}Loading Model ({model_size}) on {device.upper()}...{Style.RESET_ALL}")
        try:
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"{Fore.GREEN}✅ Model Ready on {device.upper()}!{Style.RESET_ALL}")
        except Exception as e:
            # FALLBACK MECHANISM
            if device == "cuda":
                print(f"{Fore.RED}⚠️ GPU/CUDA Error: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}🔄 Switching to CPU mode automatically...{Style.RESET_ALL}")
                try:
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    print(f"{Fore.GREEN}✅ Model Ready on CPU (Fallback)!{Style.RESET_ALL}")
                except Exception as ex_cpu:
                    logger.error(f"Critical Error on CPU: {ex_cpu}")
                    sys.exit(1)
            else:
                logger.error(f"Critical Error: {e}")
                sys.exit(1)

    def extract_audio(self, video_path: Path, audio_path: Path):
        print(f"{Fore.YELLOW}Extracting audio...{Style.RESET_ALL}")
        command = ["ffmpeg", "-i", str(video_path), "-ab", "160k", "-ac", "1", "-ar", "16000", "-vn", str(audio_path), "-y"]
        subprocess.run(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)

    def format_timestamp(self, seconds: float):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def clean_text(self, text):

        text = re.sub(r'[\(\[\{].*?[\)\]\}]', '', text)

        text = re.sub(r'[🎵🎼🎶🎸🎹🎺📢]', '', text)

        return text.strip()

    def generate_srt(self, audio_path: Path, srt_path: Path, task="transcribe", language=None):
        print(f"{Fore.YELLOW}📝 Transcribing... (Lang: {language or 'Auto'} -> Mode: {task}){Style.RESET_ALL}")
        
        prompt = "This video contains clear human speech and dialogue. Ignore music."
        
        segments, info = self.model.transcribe(
            str(audio_path), 
            beam_size=5, 
            task=task, 
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=prompt,
            condition_on_previous_text=False,
            word_timestamps=True
        )

        detected_lang = info.language.upper()
        print(f"Detected Language: {Fore.MAGENTA}{detected_lang}{Style.RESET_ALL} | Duration: {info.duration:.2f}s")

        segment_count = 0
        total_duration = info.duration
        
        with open(srt_path, "w", encoding="utf-8") as f:
            with tqdm(total=total_duration, unit="s", desc="Progress", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
                for i, segment in enumerate(segments, start=1):
                    # --- CLEANER ---
                    text = self.clean_text(segment.text)
                    
                    if not text:
                        continue
                        
                    # --- SMART TRIMMER ---
                    start_time = segment.start
                    end_time = segment.end
                    

                    if segment.words:
                        last_word_end = segment.words[-1].end

                        if end_time - last_word_end > 0.5:
                            end_time = last_word_end + 0.2

                    start_fmt = self.format_timestamp(start_time)
                    end_fmt = self.format_timestamp(end_time)
                    
                    f.write(f"{i}\n{start_fmt} --> {end_fmt}\n{text}\n\n")
                    segment_count += 1
                    
                    # Progress Bar Update
                    current_val = pbar.n
                    target_val = end_time
                    if target_val > total_duration: target_val = total_duration
                    update_amount = target_val - current_val
                    if update_amount > 0: pbar.update(update_amount)
                
                if pbar.n < total_duration: pbar.update(total_duration - pbar.n)
        
        return segment_count > 0

    def burn_subtitles(self, video_path: Path, srt_path: Path, output_path: Path, font_size=24):
        print(f"{Fore.YELLOW} Burning subtitles (Size: {font_size})...{Style.RESET_ALL}")
        
        video_str = str(video_path)
        srt_str = str(srt_path).replace("\\", "/").replace(":", "\\:")
        output_str = str(output_path)
        
        margin_v = 25
        if font_size > 30: margin_v = 40 
        
        style = f"Fontname=Arial,Fontsize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,MarginV={margin_v}"

        command = [
            "ffmpeg", "-i", video_str,
            "-vf", f"subtitles='{srt_str}':force_style='{style}'",
            "-c:a", "copy", "-y", output_str
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"{Fore.RED}FFmpeg Error:\n{result.stderr}{Style.RESET_ALL}")
                return False
            return True
        except Exception as e:
            print(f"{Fore.RED}Process Failed: {e}{Style.RESET_ALL}")
            return False

def list_videos():
    if not INPUT_DIR.exists(): INPUT_DIR.mkdir(); return []
    return [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

def main():
    if not OUTPUT_DIR.exists(): OUTPUT_DIR.mkdir()

    videos = list_videos()
    if not videos:
        print(f"{Fore.RED}No videos found in 'inputs' folder!{Style.RESET_ALL}"); return

    # --- 1. VIDEO SELECTION ---
    print(f"\n{Fore.CYAN}AVAILABLE VIDEOS:{Style.RESET_ALL}")
    for idx, video in enumerate(videos, 1): print(f"{Fore.YELLOW}{idx}.{Style.RESET_ALL} {video.name}")
    
    try:
        sel_input = input(f"Select video (1-{len(videos)}): ").strip()
        sel = int(sel_input) - 1
        selected_video = videos[sel]
    except (ValueError, IndexError):
        print("Invalid selection!"); return

    # --- 2. DEVICE SELECTION ---
    print("-" * 30)
    print(f"{Fore.CYAN}⚙️ DEVICE:{Style.RESET_ALL} [1] CPU (Default) | [2] GPU")
    dev_choice = input("Select Device: ").strip()
    
    if dev_choice == "2":
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"

    # --- 3. MODEL SELECTION ---
    model_choice = "small" 

    # --- 4. FONT SIZE ---
    print("-" * 30)
    print(f"{Fore.CYAN} STYLE SETTINGS:{Style.RESET_ALL}")
    print("Size Guide: 24 (Normal), 32 (Large/YouTube), 40 (Shorts/TikTok)")
    font_input = input("Font Size [Default: 24]: ").strip()
    try:
        font_size = int(font_input) if font_input else 24
    except ValueError:
        font_size = 24

    # --- 5. LANGUAGE SETTINGS ---
    print("-" * 30)
    custom_mode = input("Customize language? [N/y]: ").strip().lower()

    if custom_mode == 'y':
        source_lang = input("Source Lang (tr, en...): ").strip() or None
        target_lang = input("Target Lang (tr, en...): ").strip()
        
        if target_lang == "en" and source_lang != "en":
            task = "translate"
            print(f"{Fore.GREEN}>> Mode: TRANSLATE.{Style.RESET_ALL}")
        else:
            task = "transcribe"
            print(f"{Fore.GREEN}>> Mode: TRANSCRIBE.{Style.RESET_ALL}")
    else:
        source_lang = None
        task = "transcribe"
        print(f"{Fore.GREEN}>> Default Language Mode.{Style.RESET_ALL}")

    # --- PROCESS START ---
    audio_path = OUTPUT_DIR / "temp_audio.wav"
    srt_path = OUTPUT_DIR / f"{selected_video.stem}.srt"
    final_video_path = OUTPUT_DIR / f"{selected_video.stem}_subbed.mp4"

    app = VideoAutoSubtitler(model_size=model_choice, device=device, compute_type=compute_type)

    try:
        app.extract_audio(selected_video, audio_path)
        
        has_content = app.generate_srt(audio_path, srt_path, task=task, language=source_lang)
        
        if has_content:
            success = app.burn_subtitles(selected_video, srt_path, final_video_path, font_size=font_size)
            if success:
                print(f"\n{Fore.GREEN}PROCESS COMPLETE!{Style.RESET_ALL}")
                print(f"Output: {final_video_path}")
                os.startfile(OUTPUT_DIR)
        else:
            print(f"\n{Fore.RED} No speech detected.{Style.RESET_ALL}")

        if audio_path.exists(): os.remove(audio_path)

    except KeyboardInterrupt:
        print("\nProcess cancelled.")
        if audio_path.exists(): os.remove(audio_path)

if __name__ == "__main__":
    main()