import json
import os
import re


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    if "," in s_ms:
        s, ms = s_ms.split(",")
    elif "." in s_ms:
        s, ms = s_ms.split(".")
    else:
        s, ms = s_ms, 0
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(video_path):
    subtitle_path = video_path.replace("videos", "subtitles").replace(".mp4", ".srt")
    if os.path.exists(subtitle_path):
        subtitles = {}
        with open(subtitle_path, "r", encoding="utf-8") as file:
            content = file.read().split("\n\n")
            for section in content:
                if section.strip():
                    lines = section.split("\n")
                    if len(lines) >= 3:
                        time_range = lines[1].split(" --> ")
                        start_time = parse_subtitle_time(time_range[0])
                        end_time = parse_subtitle_time(time_range[1])
                        text = " ".join(line for line in lines[2:])
                        subtitles[(start_time, end_time)] = text
    else:
        subtitle_path = video_path.replace("videos", "subtitles").replace(".mp4", "_en.json")
        data_li = json.load(open(subtitle_path))
        subtitles = {}
        for dic in data_li:
            start_time = parse_subtitle_time(dic["start"])
            end_time = parse_subtitle_time(dic["end"])
            subtitles[(start_time, end_time)] = dic["line"]

    return subtitles


def extract_subtitles(video_path):
    subtitles = load_subtitles(video_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        pattern = r'<font color="white" size=".72c">(.*?)</font>'
        raw_text = re.findall(pattern, text)
        try:
            text = raw_text[0]
            subtitle_frames.append((float(start_time), float(end_time), text))
        except Exception:
            subtitle_frames.append((float(start_time), float(end_time), text))

    return subtitle_frames
