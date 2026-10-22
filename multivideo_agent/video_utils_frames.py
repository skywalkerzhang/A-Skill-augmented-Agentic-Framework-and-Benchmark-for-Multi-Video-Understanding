import os
import re
import subprocess
from pathlib import Path

import cv2
from PIL import Image

from .video_utils_base import MAX_FRAMES


def _extract_frame_with_ffmpeg(video_path: str, timestamp: float, out_path: str) -> bool:
    """Fallback single-frame extraction via ffmpeg software decode."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(timestamp)):.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-y",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if proc.returncode != 0 or not os.path.exists(out_path):
            return False
        with Image.open(out_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def extract_uniform_frames(video_path, frame_path_folder, max_frames=32):
    os.makedirs(frame_path_folder, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 {video_path}")
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    timestamp_list = []

    if total_frames <= max_frames:
        frame_indices = range(total_frames)
    else:
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp = idx / original_fps if original_fps > 0 else idx * 0.2
            frame_filename = f"frame_{timestamp:.2f}.png"
            frame_path = os.path.join(frame_path_folder, frame_filename)
            success = cv2.imwrite(frame_path, frame)
            if success:
                frame_list.append(frame_path)
                timestamp_list.append(timestamp)
            else:
                print(f"  - 保存失败: {frame_path}")
    except Exception as e:
        print(f"处理视频时出错 {video_path}: {str(e)}")
        return None, None
    finally:
        cap.release()

    sorted_pairs = sorted(zip(timestamp_list, frame_list))
    timestamp_list, frame_list = zip(*sorted_pairs) if sorted_pairs else ([], [])

    print(f"完成处理: {Path(video_path).stem}, 共保存 {len(frame_list)} 帧 (目标: {max_frames} 帧)")
    return list(frame_list), list(timestamp_list)


def timestamp_to_frames(video_path, begin_timestamp, end_timestamp, frame_folder):
    max_frames = MAX_FRAMES
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(frame_folder, video_id)
    os.makedirs(frame_folder, exist_ok=True)

    clip_frames, timestamps = [], []
    for frame in sorted(os.listdir(frame_folder), key=lambda x: float(x.split("/")[-1].split("_")[-1].split(".")[0])):
        second = float(frame.split("/")[-1].split("_")[-1].split(".")[0])
        if second > end_timestamp:
            break
        if begin_timestamp <= second <= end_timestamp:
            clip_frames.append(f"{frame_folder}/{frame}")
            timestamps.append(second)

    if len(clip_frames) > max_frames:
        chunk_size = len(clip_frames) // max_frames
        clip_frames = clip_frames[::chunk_size][:max_frames]
        timestamps = timestamps[::chunk_size][:max_frames]
    return clip_frames, timestamps


def _safe_video_id(video_path, dataset_folder):
    rel_path = video_path
    # Prefer stable key derived from path suffix after "MultiVideo_Benchmark/".
    # This avoids duplicate caches when the same dataset is accessed via different
    # mount points (e.g. /data vs /data1).
    try:
        norm = str(video_path).replace("\\", "/")
        marker = "/MultiVideo_Benchmark/"
        if marker in norm:
            rel_path = norm.split(marker, 1)[1]
    except Exception:
        pass

    # Make cache keys stable across different mounts/symlinks (e.g. /data vs /data1)
    # by comparing real paths.
    try:
        vp_real = os.path.realpath(video_path)
    except Exception:
        vp_real = video_path
    try:
        ds_real = os.path.realpath(dataset_folder) if dataset_folder else ""
    except Exception:
        ds_real = dataset_folder or ""

    if dataset_folder:
        # Prefer relpath when video is under dataset_folder (after resolving symlinks).
        if ds_real and vp_real.startswith(ds_real.rstrip(os.sep) + os.sep):
            rel_path = os.path.relpath(vp_real, ds_real)
        elif video_path.startswith(dataset_folder.rstrip(os.sep) + os.sep):
            rel_path = os.path.relpath(video_path, dataset_folder)
    rel_path = rel_path.lstrip(os.sep)
    rel_path = os.path.splitext(rel_path)[0]
    rel_path = rel_path.replace(os.sep, "__")
    rel_path = re.sub(r"[^A-Za-z0-9._-]+", "_", rel_path)
    return rel_path


def timestamp_to_clip_path(dataset_folder, begin_time_stamp, end_time_stamp, video_path, fps=2):
    max_frames = MAX_FRAMES
    video_id = _safe_video_id(video_path, dataset_folder)
    frame_folder = os.path.join(dataset_folder, f"dense_frames/{video_id}/")
    os.makedirs(frame_folder, exist_ok=True)

    # 如果输入是帧目录（例如 Forensic_Detection 的 png 序列），直接从目录采样帧。
    # 这里用一个“虚拟时长”做映射，避免不同任务因缺少真实时长而出现 empty segment。
    if os.path.isdir(video_path):
        exts = (".png", ".jpg", ".jpeg", ".webp")
        files = [f for f in os.listdir(video_path) if f.lower().endswith(exts)]
        if not files:
            return [], []
        # 尽量按数字排序（如 1162.png），否则按字典序
        def _key(x: str):
            base = os.path.splitext(x)[0]
            try:
                return (0, int(base))
            except Exception:
                return (1, base)
        files = sorted(files, key=_key)

        total = len(files)
        virtual_duration = 100.0
        b = float(begin_time_stamp)
        e = float(end_time_stamp)
        if e <= b:
            e = b + 1.0
        b = max(0.0, min(b, virtual_duration))
        e = max(0.0, min(e, virtual_duration))
        if e <= b:
            e = min(virtual_duration, b + 1.0)

        start_idx = int((b / virtual_duration) * total)
        end_idx = int((e / virtual_duration) * total)
        start_idx = max(0, min(start_idx, total - 1))
        end_idx = max(start_idx + 1, min(end_idx, total))
        chosen = files[start_idx:end_idx]
        # 控制帧数上限
        if len(chosen) > max_frames:
            step = max(1, len(chosen) // max_frames)
            chosen = chosen[::step][:max_frames]
        frame_paths = [os.path.join(video_path, f) for f in chosen]
        # 生成近似 timestamps（均匀分布）
        if len(frame_paths) == 1:
            timestamps = [b]
        else:
            span = max(1e-6, (e - b))
            timestamps = [b + span * (i / (len(frame_paths) - 1)) for i in range(len(frame_paths))]
        return frame_paths, timestamps

    if (end_time_stamp - begin_time_stamp) < 1:
        begin_time_stamp = max(begin_time_stamp - 0.5, 0)
        end_time_stamp += 0.5

    if int((end_time_stamp - begin_time_stamp) * fps) < max_frames:
        num_frames = int((end_time_stamp - begin_time_stamp) * fps)
        time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(num_frames)]
    else:
        num_frames = max_frames
        time_points = [begin_time_stamp + i * ((end_time_stamp - begin_time_stamp) / num_frames) for i in range(num_frames)]

    cap = cv2.VideoCapture(video_path)
    cap_opened = cap.isOpened()
    if not cap_opened:
        # Let ffmpeg fallback handle extraction below.
        video_fps = 0.0
        frame_count = 0.0
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    # 用真实视频时长裁剪 begin/end，避免请求越界导致 empty segment
    if video_fps > 0 and frame_count > 0:
        duration = float(frame_count) / float(video_fps)
        begin_time_stamp = max(0.0, float(begin_time_stamp))
        end_time_stamp = float(end_time_stamp)
        # 如果请求起点越界：回退到视频末尾一段，避免 empty segment
        if begin_time_stamp >= duration:
            window = max(1.0, min(20.0, end_time_stamp - begin_time_stamp if end_time_stamp > begin_time_stamp else 20.0))
            begin_time_stamp = max(0.0, duration - window)
            end_time_stamp = duration
        else:
            end_time_stamp = min(end_time_stamp, duration)

    frame_paths = []
    timestamps = []

    for t in time_points:
        frame_name = f"frame_{t:.2f}.png"
        frame_path = os.path.join(frame_folder, frame_name)

        valid_frame = False
        if os.path.exists(frame_path):
            try:
                with Image.open(frame_path) as img:
                    img.verify()
                valid_frame = True
            except Exception:
                try:
                    os.remove(frame_path)
                except Exception:
                    pass

        if not valid_frame and cap_opened and video_fps > 0:
            frame_idx = int(round(t * video_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if success:
                cv2.imwrite(frame_path, frame)
                valid_frame = True
            else:
                # AV1/hardware decode failure fallback.
                valid_frame = _extract_frame_with_ffmpeg(video_path, t, frame_path)
        elif not valid_frame:
            # cv2 cannot open/read: fallback directly to ffmpeg software decode.
            valid_frame = _extract_frame_with_ffmpeg(video_path, t, frame_path)

        if valid_frame:
            try:
                with Image.open(frame_path) as img:
                    img.verify()
                frame_paths.append(frame_path)
                timestamps.append(t)
            except Exception:
                continue

    cap.release()

    if len(frame_paths) > max_frames:
        chunk_size = len(frame_paths) // max_frames
        frame_paths = frame_paths[::chunk_size][:max_frames]
        timestamps = timestamps[::chunk_size][:max_frames]

    return frame_paths, timestamps


def clip_number_to_clip_path(dataset_folder, clip_numbers, video_path, clip_duration=10, fps=2):
    video_id = _safe_video_id(video_path, dataset_folder)
    frame_folder = os.path.join(dataset_folder, "dense_frames", video_id)
    os.makedirs(frame_folder, exist_ok=True)

    frame_list = []
    second_list = []

    cap = cv2.VideoCapture(video_path)
    cap_opened = cap.isOpened()
    video_fps = (cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap_opened else 0.0

    for clip_number in clip_numbers:
        begin_time_stamp = clip_number * clip_duration
        end_time_stamp = begin_time_stamp + clip_duration
        time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(int(clip_duration * fps))]

        for t in time_points:
            frame_name = f"frame_{t:.2f}.png"
            frame_path = os.path.join(frame_folder, frame_name)

            valid_frame = False
            if os.path.exists(frame_path):
                try:
                    with Image.open(frame_path) as img:
                        img.verify()
                    valid_frame = True
                except Exception:
                    try:
                        os.remove(frame_path)
                    except Exception:
                        pass

            if not valid_frame and cap_opened and video_fps > 0:
                frame_idx = int(round(t * video_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if success:
                    cv2.imwrite(frame_path, frame)
                    valid_frame = True
                else:
                    valid_frame = _extract_frame_with_ffmpeg(video_path, t, frame_path)
            elif not valid_frame:
                valid_frame = _extract_frame_with_ffmpeg(video_path, t, frame_path)

            if valid_frame:
                try:
                    with Image.open(frame_path) as img:
                        img.verify()
                    frame_list.append(frame_path)
                    second_list.append(t)
                except Exception as e:
                    print(f"Error at time {t:.3f}: {e}")
                    continue

    cap.release()

    if len(frame_list) > MAX_FRAMES:
        interval = len(frame_list) / MAX_FRAMES
        frame_list = [frame_list[int(i * interval)] for i in range(MAX_FRAMES)]
        second_list = [second_list[int(i * interval)] for i in range(MAX_FRAMES)]

    if len(frame_list) == 0:
        raise KeyError(
            f"Frame list {dataset_folder} {clip_numbers} {video_path} {clip_duration} {fps} is invalid!"
        )

    return frame_list, second_list
