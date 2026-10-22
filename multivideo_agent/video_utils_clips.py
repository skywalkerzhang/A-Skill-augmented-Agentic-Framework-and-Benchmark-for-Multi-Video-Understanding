import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from decord import VideoReader, cpu

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except Exception:
    from moviepy import VideoFileClip, concatenate_videoclips

from .video_utils_base import _seconds_to_time_str


def custom_ffmpeg_extract_subclip(filename, t1, t2, targetname):
    flag, retry = 0, 0

    while flag != 1 and retry < 5:
        folder = os.path.dirname(targetname)
        os.makedirs(folder, exist_ok=True)

        try:
            video = VideoFileClip(filename)
            video_sub_clip = video.subclip(t1, t2)

            width, height = video.size
            scale = min(700 / max(width, height), 1)
            if scale < 1:
                video_sub_clip = video_sub_clip.resize(scale)

            video_sub_clip = concatenate_videoclips([video_sub_clip])
            video_sub_clip.write_videofile(targetname, codec="libx264", audio_codec="aac", logger=None)

            VideoReader(targetname, ctx=cpu(0), num_threads=1)
            flag = 1
        except Exception as e:
            print(f"Attempt {retry + 1} failed: {e}")
            retry += 1

    if not flag:
        print(f"Failed to process {targetname} after {retry} retries.")


def extract_video_clip(video_path: str, start: float, end: float) -> str:
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        start = max(0, min(start, duration))
        end = max(0, min(end, duration))
        if end <= start:
            start = end - 1

        subclip = clip.subclip(start, end)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        subclip.write_videofile(tmp_path, codec="libx264", audio_codec="aac", logger=None)
    return tmp_path


def _get_video_duration(video_path):
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
    return duration


def _cut_video_clips(video_path, clips_dir, video_name, duration):
    existing_clips = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    expected_clips = int(duration // 10) + (1 if duration % 10 > 0 else 0)
    if len(existing_clips) >= expected_clips - 2:
        print(f"Skipping cutting: {len(existing_clips)} clips already exist in {clips_dir}")
        return

    def cut_single_clip(start_time, end_time, clip_index):
        start_str = _seconds_to_time_str(start_time)
        end_str = _seconds_to_time_str(end_time)

        output_file = os.path.join(
            clips_dir,
            f"clip_{clip_index}_{start_str.replace(':', '-')}_to_{end_str.replace(':', '-')}.mp4",
        )

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-ss",
            str(start_time),
            "-t",
            str(end_time - start_time),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            output_file,
            "-y",
        ]

        subprocess.run(cmd, capture_output=True)
        print(f"Created clip: {os.path.basename(output_file)}")

    clip_duration = 10
    num_clips = int(duration // clip_duration) + (1 if duration % clip_duration > 0 else 0)

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for i in range(num_clips):
            start_time = i * clip_duration
            end_time = min((i + 1) * clip_duration, duration)
            futures.append(executor.submit(cut_single_clip, start_time, end_time, i))

        for future in as_completed(futures):
            future.result()
