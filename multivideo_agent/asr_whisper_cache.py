import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List

from openai import OpenAI


def normalize_task_name(task: Any) -> str:
    if isinstance(task, list):
        task = task[0] if task else ""
    return str(task or "").strip()


def is_defeasible_entailment_task(task: Any) -> bool:
    compact = re.sub(r"[^a-z]", "", normalize_task_name(task).lower())
    return compact == "defeasibleentailment"


def task_slug(task: Any) -> str:
    name = normalize_task_name(task)
    if not name:
        return "unknown_task"
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "unknown_task"


def _format_asr_segments(segments: List[Dict[str, Any]], fallback_text: str) -> str:
    rows: List[str] = []
    for seg in segments or []:
        try:
            start = int(float(seg.get("start", 0.0)))
            end = int(float(seg.get("end", start)))
            text = str(seg.get("text", "")).strip()
        except Exception:
            continue
        if text:
            rows.append(f"{start}-{max(end, start)}:{text}")
    if rows:
        return " ".join(rows)
    fallback_text = (fallback_text or "").strip()
    return f"0-0:{fallback_text}" if fallback_text else ""


def cache_path_for_video(
    video_path: str,
    cache_dir: str,
    model_name: str,
    task_name: Any = "Defeasible Entailment",
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    try:
        st = os.stat(video_path)
        sig = f"{os.path.realpath(video_path)}::{int(st.st_size)}::{int(st.st_mtime)}::{model_name}"
    except Exception:
        sig = f"{os.path.realpath(video_path)}::{model_name}"
    digest = hashlib.md5(sig.encode("utf-8")).hexdigest()
    prefix = task_slug(task_name)
    return os.path.join(cache_dir, f"{prefix}_{digest}.json")


def transcribe_video_openai_whisper_cached(
    video_path: str,
    *,
    api_base: str,
    api_key: str,
    model_name: str = "whisper-1",
    cache_dir: str = "./model_cache/asr",
    task_name: Any = "Defeasible Entailment",
) -> str:
    cache_path = cache_path_for_video(
        video_path=video_path,
        cache_dir=cache_dir,
        model_name=model_name,
        task_name=task_name,
    )
    if os.path.exists(cache_path):
        try:
            data = json.load(open(cache_path, "r", encoding="utf-8"))
            cached = str(data.get("subtitles", "")).strip()
            if cached:
                return cached
        except Exception:
            pass

    if not api_key:
        return ""

    client = OpenAI(base_url=api_base, api_key=api_key)
    subtitles = ""
    try:
        with open(video_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model_name,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        if isinstance(resp, dict):
            text = str(resp.get("text", "") or "")
            segments = resp.get("segments", []) or []
        else:
            text = str(getattr(resp, "text", "") or "")
            segments = getattr(resp, "segments", []) or []
            if not isinstance(segments, list):
                segments = []
        subtitles = _format_asr_segments(segments, text)
    except Exception:
        subtitles = ""

    if not subtitles:
        try:
            with open(video_path, "rb") as f:
                resp_text = client.audio.transcriptions.create(
                    model=model_name,
                    file=f,
                    response_format="text",
                )
            raw_text = str(resp_text or "").strip()
            subtitles = f"0-0:{raw_text}" if raw_text else ""
        except Exception:
            return ""

    payload = {
        "task": normalize_task_name(task_name),
        "video_path": video_path,
        "model": model_name,
        "api_base": api_base,
        "subtitles": subtitles,
        "updated_at": int(time.time()),
    }
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as wf:
        json.dump(payload, wf, ensure_ascii=False)
    os.replace(tmp, cache_path)
    return subtitles
