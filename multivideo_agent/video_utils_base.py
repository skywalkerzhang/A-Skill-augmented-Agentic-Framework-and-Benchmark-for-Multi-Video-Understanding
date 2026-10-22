import ast
import re

import cv2
import decord
from decord import VideoReader

MAX_FRAMES = 4
MAX_RESOLUTION = 1024
MAX_DS_ROUND = 20


def robust_eval(s: str):
    if not isinstance(s, str):
        return s

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    s_fixed = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s_fixed = re.sub(r",\s*\.", ",", s_fixed)
    s_fixed = re.sub(r"\)\s*\(", "),(", s_fixed)

    open_paren_count = s_fixed.count("(")
    close_paren_count = s_fixed.count(")")
    if open_paren_count > close_paren_count:
        s_fixed += ")" * (open_paren_count - close_paren_count)

    double_quote_count = s_fixed.count('"')
    if double_quote_count % 2 == 1:
        s_fixed += '"'

    def fix_nested_strings(text):
        pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*"([^"]*)$'

        def replacer(match):
            num1, num2, incomplete_desc = match.groups()
            return f'({num1}, {num2}, "{incomplete_desc}")'

        return re.sub(pattern, replacer, text)

    s_fixed = fix_nested_strings(s_fixed).strip()
    if s_fixed.startswith("(") and not s_fixed.startswith("["):
        s_fixed = "[" + s_fixed + "]"

    try:
        return ast.literal_eval(s_fixed)
    except Exception:
        try:
            s_fixed = re.sub(r",\s*\.\s*,", ",", s_fixed)
            s_fixed = re.sub(r",\s*\.\s*\]", "]", s_fixed)
            s_fixed = re.sub(r",\s*\.\s*$", "", s_fixed)
            return ast.literal_eval(s_fixed)
        except Exception:
            return s_fixed


def is_valid_video(path):
    try:
        cap = cv2.VideoCapture(path)
    except Exception:
        return False

    if not cap.isOpened():
        return False

    try:
        VideoReader(path, num_threads=1)
        return True
    except Exception:
        return False


def _seconds_to_time_str(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
