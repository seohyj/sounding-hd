import csv
import os
import yt_dlp
import time

CSV_FILE = os.path.join(os.getenv('DATA_ROOT', './data'), 'mrhisum/metadata.csv')
DOWNLOAD_PATH = os.path.join(os.getenv('DATA_ROOT', './data'), 'mrhisum/video_raw')
POSSIBLE_EXTENSIONS = ['.mp4', '.mkv', '.webm']

os.makedirs(DOWNLOAD_PATH, exist_ok=True)

print(f"Reading CSV file: {CSV_FILE}")
print(f"Videos will be saved to: {DOWNLOAD_PATH}\n")

try:
    with open(CSV_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            youtube_id = row['youtube_id'].strip()
            video_id = row['video_id'].strip()

            if not youtube_id or not video_id:
                print(f"Skipping row {i+1} with empty youtube_id or video_id.")
                continue

            video_url = f"https://www.youtube.com/watch?v={youtube_id}"

            output_base = os.path.join(DOWNLOAD_PATH, video_id)

            file_exists = False
            found_file_path = ""
            for ext in POSSIBLE_EXTENSIONS:
                file_path = output_base + ext
                if os.path.exists(file_path):
                    file_exists = True
                    found_file_path = file_path
                    break

            if file_exists:
                continue

            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_base,
                'quiet': True,
                'no_warnings': True,
            }

            print(f"[START] Downloading: {video_id} (URL: {video_url})")

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                print(f"[SUCCESS] Finished downloading: {video_id}\n")

            except yt_dlp.utils.DownloadError as e:
                print(f"[ERROR] Failed to download {video_id}: {e}\n")
            except Exception as e:
                print(f"[ERROR] An unexpected error occurred for {video_id}: {e}\n")

except FileNotFoundError:
    print(f"ERROR: CSV file not found at '{CSV_FILE}'")
except Exception as e:
    print(f"An critical error occurred: {e}")

print("All download attempts finished.")
