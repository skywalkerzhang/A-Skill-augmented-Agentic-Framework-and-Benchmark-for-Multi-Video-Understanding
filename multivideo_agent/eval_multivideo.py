# =============================================================================
# 环境变量配置（必须在导入其他库之前设置）
# =============================================================================
import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_MODULE_LOADING"] = "EAGER"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_USE_MODELSCOPE"] = "false"
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =============================================================================
# 标准库导入
# =============================================================================
import sys
import re
import json
import time
import random
import hashlib
import pickle
import fcntl
import argparse
import traceback
import logging
import faulthandler
import glob
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from difflib import SequenceMatcher
import multiprocessing as mp

# 设置 multiprocessing 启动方式
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass

faulthandler.enable(all_threads=True)

# =============================================================================
# 第三方库导入
# =============================================================================
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# 设置随机种子（可复现性）
GLOBAL_SEED = int(os.getenv("SEED", "42"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      # 关闭自动调优，保证确定性

# =============================================================================
# 本地模块导入
# =============================================================================
from .video_utils import (
    _get_video_duration, _cut_video_clips, extract_subtitles,
    timestamp_to_clip_path, is_valid_video, is_valid_frame,
    extract_video_clip, parse_subtitle_time, clip_number_to_clip_path,
    image_paths_to_base64, load_image, robust_eval, timestamp_to_frames
)
from .retriever_languagebind import Retrieval_Manager
from .prompt import *
from .skills import format_skills_for_prompt
from .asr_whisper_cache import (
    is_defeasible_entailment_task,
    transcribe_video_openai_whisper_cached,
)
from .vision_tools import (
    detect_scene_changes,
    run_object_detection,
    summarize_object_detection,
    run_spatial_relations,
    summarize_spatial_relations,
    run_scene_graph,
    run_scene_graph_grounded,
    summarize_scene_graph_results,
    visual_similarity_from_embeddings,
    visual_similarity_from_frames,
    _image_embeddings,
)

# =============================================================================
# 日志配置
# =============================================================================
_base_dir = Path(__file__).resolve().parent.parent
_logs_dir = _base_dir / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_log_ts = time.strftime("%Y%m%d_%H%M%S")
_log_path = _logs_dir / f"debug_run_{_log_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_log_path), mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("DEBUG")

logger.info(
    f"[REPRODUCIBILITY] seed={GLOBAL_SEED}, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED','?')}, "
    f"cudnn.deterministic={torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'}, "
    f"cudnn.benchmark={torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A'}, "
    f"allow_tf32={torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else 'N/A'}"
)

# =============================================================================
# 常量定义
# =============================================================================
MAX_DS_ROUND = int(os.getenv("MAX_DS_ROUND", "5"))
MIN_QUESTION_LENGTH = 10
DEFAULT_VIDEO_DURATION = 100.0
DEFAULT_VLM_MAX_TOKENS = 2048
DEFAULT_CLIP_FPS = 2.0
MAX_IMAGE_DIM = 720
API_RETRY_COUNT = 3
API_RETRY_DELAY = 5
VIDEO_IDS = ("ref", "A", "B", "C", "D")
ANSWER_LIST = ("A", "B", "C", "D")
ANSWER_CHOICES = set(ANSWER_LIST)
RE_VIDEO_READER_ID = re.compile(r'<video_reader id="([^"]+)">')
RE_TG_AGENT = re.compile(r'<temporal_grounding_agent id="([^"]+)">([\s\S]*?)</temporal_grounding_agent>')
RE_SEGMENT_QUERY = re.compile(r'<video_segment_retriever_textual_query id="([^"]+)">([\s\S]*?)</video_segment_retriever_textual_query>')
RE_SUBTITLE_RETRIEVER = re.compile(r'<subtitle_retriever id="([^"]+)">([^<]+)</subtitle_retriever>')
RE_SUBTITLE_EXTRACTOR = re.compile(r'<subtitle_extractor id="([^"]+)">([^<]+)</subtitle_extractor>')
RE_VIDEO_BROWSER = re.compile(r'<video_browser id="([^"]+)">([^<]+)</video_browser>')
RE_VIDEO_READER = re.compile(r'<video_reader id="([^"]+)">([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>')
RE_SCENE_DETECTOR = re.compile(r'<scene_detector id="([^"]+)">([^<]+)</scene_detector>')
RE_OBJECT_TRACKER = re.compile(r'<object_tracker id="([^"]+)">([^<]+)</object_tracker>')
RE_SPATIAL_RELATION = re.compile(r'<spatial_relation id="([^"]+)">([^<]+)</spatial_relation>')
RE_SCENE_GRAPH = re.compile(r'<scene_graph id="([^"]+)">([^<]+)</scene_graph>')
RE_VISUAL_SIMILARITY = re.compile(r'<visual_similarity id="([^"]+)">([^<]+)</visual_similarity>')
RE_REMOVE_CODE_BLOCK = re.compile(r"```.*?\n", flags=re.S)
RE_TAG_ANSWER = re.compile(r"<answer>(.*?)</answer>", flags=re.S | re.I)
RE_TAG_THINK = re.compile(r"<think>.*?</think>", flags=re.S | re.I)
RE_JSON_ANSWER = re.compile(r'["\'](answer|final)["\'].*?["\']\s*([A-Z])\s*["\']', flags=re.I)
RE_PATTERN_ANSWER = re.compile(r"(final\s*answer|best\s*option|answer)\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?", flags=re.I)
RE_PATTERN_OPTION = re.compile(r"\b(option|choice|select)\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?\b", flags=re.I)
RE_PATTERN_VIDEO = re.compile(r"\b(video|candidate)\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?\b", flags=re.I)

# =============================================================================
# 属性冲突检测框架
# Detection tools (scene_graph, object_tracker) 通过像素级分析和 bounding-box
# 检测，对**可度量视觉属性**比 VLM 更可靠：
#   可信度高：颜色、物体数量、物体类型、位置、大小
# VLM 更擅长：
#   动作、时序事件、上下文语义、细粒度描述
# =============================================================================
CONFLICT_REREAD = "llm"           # 冲突时引导 planner 自己 re-read，不硬编码工具调用
ENABLE_ATTRIBUTE_OVERRIDE = True  # 是否用 detection 工具参与冲突检测
                                  # 提高颜色冲突阈值以降低误改

_SIGNAL_COLORS_SET = frozenset(('red', 'green', 'yellow', 'orange'))
_SIGNAL_COLOR_RE = re.compile(r'\b(red|green|yellow|orange)\b', re.I)
_NUMBER_WORDS = {
    'zero': 0, 'no': 0, 'one': 1, 'a ': 1, 'two': 2, 'three': 3,
    'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine': 9, 'ten': 10,
}

TRUST_GUIDE = (
    "📌 Conflict detected between VLM and detection tools. "
    "Detection is more reliable for: object count, presence, coarse position. "
    "VLM is more reliable for: semantics, state, action, texture, context. "
    "A re-read has been triggered. Use the re-read result to resolve."
)


def _extract_corrected_color(text: str) -> str:
    """Extract corrected color if present, otherwise first color mention."""
    if not text:
        return ""
    m = re.search(r'color corrected:\s*([a-z]+)', text, re.I)
    if m:
        return m.group(1).lower()
    m = _SIGNAL_COLOR_RE.search(text)
    return m.group(1).lower() if m else ""


# ---------- 辅助：从 detection tool 证据中解析信号色 profile ----------
def _det_signal_color_profile(evidence: str):
    """Parse scene_graph / object_tracker evidence for signal colors.

    Returns ``(dominant_color, n_frames, total_frames, total_objects)``.
    Each "t=X.Xs:" line counts as one frame.
    """
    frame_counts: Dict[str, List[int]] = defaultdict(list)
    total_frames = 0
    for line in evidence.strip().splitlines():
        line = line.strip()
        if not re.match(r't=[\d.]+s:', line):
            continue
        total_frames += 1
        seen: set = set()
        # "4 green", "1 orange" …
        for m in re.finditer(r'(\d+)\s+(red|green|yellow|orange)\b', line, re.I):
            c = m.group(2).lower()
            cnt = int(m.group(1))
            if c in seen:
                frame_counts[c][-1] += cnt
            else:
                frame_counts[c].append(cnt)
                seen.add(c)
        # "green small traffic light" (implicit 1)
        for m in re.finditer(r'\b(red|green|yellow|orange)\s+(?:small|medium|large)', line, re.I):
            c = m.group(1).lower()
            if c not in seen:
                frame_counts[c].append(1)
                seen.add(c)
    if not frame_counts:
        return None, 0, total_frames, 0
    ranked = sorted(frame_counts.items(),
                    key=lambda x: (len(x[1]), sum(x[1])), reverse=True)
    best, counts = ranked[0]
    second_frames = len(ranked[1][1]) if len(ranked) > 1 else 0
    # 仅当 best 是明确赢家时返回（帧数 > 第二名，或帧数相同但物体数多 2 倍以上）
    if len(counts) < second_frames:
        return None, 0, total_frames, 0   # 没有明确赢家
    return best, len(counts), total_frames, sum(counts)


# ---------- 辅助：从 detection tool 证据中解析物体计数 ----------
def _det_object_counts(evidence: str) -> Dict[str, List[int]]:
    """Parse detection evidence for object counts per label.

    Returns ``{label: [count_per_frame, ...]}``.
    """
    counts: Dict[str, List[int]] = defaultdict(list)
    for line in evidence.strip().splitlines():
        line = line.strip()
        if not re.match(r't=[\d.]+s:', line):
            continue
        # "6 traffic light", "3 person" etc.
        for m in re.finditer(r'(\d+)\s+([a-z][a-z ]+?)(?:\s*\(|,|\[|$)', line, re.I):
            label = m.group(2).strip().lower()
            counts[label].append(int(m.group(1)))
        # "small traffic light" (implicit 1)
        if re.search(r'(?:small|medium|large)\s+\w', line, re.I):
            m2 = re.search(r'(?:small|medium|large)\s+([\w ]+?)(?:\s*\(|$)', line, re.I)
            if m2:
                label = m2.group(1).strip().lower()
                counts[label].append(1)
    return dict(counts)


def _vlm_extract_count(vlm_text: str, label: str) -> Optional[int]:
    """Try to extract the count of *label* from VLM free-text.

    Returns int if found, else None.  Very conservative to avoid false positives.
    """
    ll = label.lower()
    text = vlm_text.lower()
    if ll not in text:
        return None
    # "no traffic lights" / "no traffic light"
    if re.search(r'\bno\s+' + re.escape(ll), text):
        return 0
    # "3 traffic lights", "three traffic lights"
    m = re.search(r'(\d+)\s+' + re.escape(ll), text)
    if m:
        return int(m.group(1))
    for word, val in _NUMBER_WORDS.items():
        if re.search(re.escape(word) + r'\s+' + re.escape(ll), text):
            return val
    # "a traffic light" → 1
    if re.search(r'\ba\s+' + re.escape(ll), text):
        return 1
    return None


# ---------- 通用冲突检测入口 ----------
def _detect_attribute_conflicts(
    vlm_text: str,
    sg_evidence: str,
    ot_evidence: str = "",
) -> List[Dict[str, Any]]:
    """Detect measurable-attribute conflicts between VLM and detection tools.

    Returns list of dicts, each with keys:
        type, vlm_val, det_val, confidence, note
    """
    conflicts: List[Dict[str, Any]] = []

    # ── 1) Signal-color conflict ──
    vlm_color = None
    m = _SIGNAL_COLOR_RE.search(vlm_text)
    if m:
        vlm_color = m.group(1).lower()

    combined_ev = (sg_evidence + "\n" + ot_evidence).strip()
    sg_color, sg_nf, sg_tot, sg_obj = _det_signal_color_profile(sg_evidence)
    ot_color, ot_nf, ot_tot, ot_obj = _det_signal_color_profile(ot_evidence)

    det_color = sg_color
    confidence = sg_nf / max(sg_tot, 1) if sg_tot else 0.0
    ot_agrees = False
    if ot_color and ot_color == sg_color:
        ot_agrees = True
        confidence = min(1.0, confidence + 0.2)
    elif not det_color and ot_color and ot_nf >= 1:
        det_color = ot_color
        confidence = 0.4

    if vlm_color and det_color and vlm_color != det_color and confidence >= 0.60:
        note = f"scene_graph={det_color} ({sg_nf}/{sg_tot} frames, {sg_obj} objects)"
        if ot_agrees:
            note += "; object_tracker agrees"
        conflicts.append({
            'type': 'color',
            'vlm_val': vlm_color,
            'det_val': det_color,
            'confidence': confidence,
            'note': note,
        })

    # ── 2) Object-count conflict ──
    det_counts_sg = _det_object_counts(sg_evidence)
    det_counts_ot = _det_object_counts(ot_evidence)
    # 合并两个来源，取最大中位数
    merged_labels: set = set(det_counts_sg) | set(det_counts_ot)
    for label in merged_labels:
        sg_vals = det_counts_sg.get(label, [])
        ot_vals = det_counts_ot.get(label, [])
        if not sg_vals and not ot_vals:
            continue
        # 用中位数作为 detection 估计值
        all_vals = sg_vals + ot_vals
        det_median = sorted(all_vals)[len(all_vals) // 2]
        if det_median == 0:
            continue
        vlm_cnt = _vlm_extract_count(vlm_text, label)
        if vlm_cnt is None:
            continue  # VLM 没提到这个 label 的数量，不算冲突
        # 只在差距显著时报告 (>50% 或差 ≥3)
        diff = abs(vlm_cnt - det_median)
        if diff >= 3 or (det_median > 0 and diff / det_median > 0.5):
            n_frames = len(sg_vals) + len(ot_vals)
            conflicts.append({
                'type': 'count',
                'vlm_val': f"{vlm_cnt} {label}",
                'det_val': f"~{det_median} {label}",
                'confidence': min(1.0, n_frames / 6),
                'note': f"detection tools report ~{det_median} across {n_frames} frames",
            })

    # ── 3) Object-presence conflict: VLM says "no X" but detection found X ──
    for label, vals in {**det_counts_sg, **det_counts_ot}.items():
        if not vals or max(vals) < 2:
            continue  # 检测不够强
        vlm_cnt = _vlm_extract_count(vlm_text, label)
        if vlm_cnt == 0:
            n_frames = len(vals)
            conflicts.append({
                'type': 'presence',
                'vlm_val': f"no {label}",
                'det_val': f"{label} detected ({max(vals)} in best frame)",
                'confidence': min(1.0, n_frames / 4),
                'note': f"detection found {label} in {n_frames} frames but VLM says none",
            })

    return conflicts
RE_NA = re.compile(r"\bN\s*/?\s*A\b", flags=re.I)
RE_UNCERTAIN = re.compile(
    r"\b("
    r"not sure|uncertain|unsure|maybe|"
    r"can't tell|cannot tell|can't determine|cannot determine|impossible to determine|"
    r"hard to tell|insufficient|unclear|"
    r"not visible|not clearly visible|not clearly seen|no clear visibility|"
    r"out of frame|not in frame|"
    # 明确表示缺失/看不到的表述
    r"does not show|no consistent|no clear|cannot confirm|"
    r"not discernible|indistinguishable|ambiguous"
    r")\b",
    flags=re.I,
)
# =============================================================================
# 工具函数
# =============================================================================

# 结构化字段名集合（CAPTION 以外的所有字段都归入 evidence）
# 注意：OBJECTS/ATTRIBUTES 已从 prompt 中移除（VLM 幻觉严重），
# 但解析器仍兼容旧输出以防残留。
_STRUCTURED_FIELD_NAMES = ("ACTIONS", "SPATIAL", "EVIDENCE", "ANSWER", "OBJECTS", "ATTRIBUTES")
_STRUCTURED_FIELDS = re.compile(
    r"(?i)^(" + "|".join(_STRUCTURED_FIELD_NAMES) + r")\s*:", re.MULTILINE
)
_ALL_FIELD_NAMES_PATTERN = "|".join(("CAPTION",) + _STRUCTURED_FIELD_NAMES)

def _split_caption_evidence(text: str) -> tuple[str, str]:
    """Parse VLM output into (caption, evidence).

    Supports both the old 2-line format and the new structured multi-field format:
        CAPTION: ...
        ACTIONS: ...
        SPATIAL: ...
        EVIDENCE: ...
    All fields except CAPTION are concatenated into the evidence string.
    Falls back gracefully if format not found.
    """
    if not text:
        return "", ""
    t = text.strip()

    # ── 结构化格式：包含 ACTIONS/SPATIAL/EVIDENCE 等字段 ──
    has_structured = bool(_STRUCTURED_FIELDS.search(t))
    has_caption = bool(re.search(r"(?i)\bcaption\s*:", t))

    if has_caption and has_structured:
        # 提取 CAPTION
        cap_match = re.search(
            r"(?i)\bcaption\s*:\s*(.*?)(?=\n\s*(?:" + _ALL_FIELD_NAMES_PATTERN + r")\s*:|$)",
            t, re.DOTALL,
        )
        caption = cap_match.group(1).strip() if cap_match else ""

        # 提取所有非 CAPTION 字段，保留字段名作为结构化标记
        evidence_parts = []
        for m in re.finditer(
            r"(?i)\b(" + "|".join(_STRUCTURED_FIELD_NAMES) + r")\s*:\s*(.*?)"
            r"(?=\n\s*(?:" + _ALL_FIELD_NAMES_PATTERN + r")\s*:|$)",
            t, re.DOTALL,
        ):
            field_name = m.group(1).upper()
            field_val = m.group(2).strip()
            if field_val:
                evidence_parts.append(f"{field_name}: {field_val}")
        evidence = "\n".join(evidence_parts)
        return caption, evidence or t

    # ── 旧格式兼容：CAPTION + EVIDENCE (2-line) ──
    m_cap = re.search(r"(?i)\bcaption\s*:\s*(.*)", t)
    m_evi = re.search(r"(?i)\bevidence\s*:\s*(.*)", t)
    m_ans = re.search(r"(?i)\banswer\s*:\s*(.*)", t)  # legacy
    if not (m_cap or m_evi or m_ans):
        return "", t
    caption = ""
    evidence = ""
    if "CAPTION:" in t.upper() and "EVIDENCE:" in t.upper():
        parts = re.split(r"(?i)\bevidence\s*:\s*", t, maxsplit=1)
        before = parts[0]
        after = parts[1] if len(parts) > 1 else ""
        caption = re.sub(r"(?i)\bcaption\s*:\s*", "", before).strip()
        evidence = after.strip()
        return caption, evidence
    if "CAPTION:" in t.upper() and "ANSWER:" in t.upper() and "EVIDENCE:" not in t.upper():
        parts = re.split(r"(?i)\banswer\s*:\s*", t, maxsplit=1)
        before = parts[0]
        after = parts[1] if len(parts) > 1 else ""
        caption = re.sub(r"(?i)\bcaption\s*:\s*", "", before).strip()
        evidence = after.strip()
        return caption, evidence
    if m_cap:
        caption = m_cap.group(1).strip()
    if m_evi:
        evidence = m_evi.group(1).strip()
    elif m_ans:
        evidence = m_ans.group(1).strip()
    return caption, evidence or t

def safe_write_with_lock(data, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_tool_params(text: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if not text:
        return params
    parts = [p.strip() for p in re.split(r"[;|,]", text) if p.strip()]
    positional = []
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().lower()] = value.strip()
        else:
            positional.append(part)
    if positional:
        params["_positional"] = positional
    # 兼容 planner 的错误格式：start=0:end=13 或 start=0;end=13
    if "range" not in params:
        start_val = params.get("start")
        end_val = params.get("end")
        if start_val and end_val:
            params["range"] = f"{start_val}:{end_val}"
        elif start_val and "end=" in str(start_val):
            m = re.match(r"\s*([0-9.]+)\s*:\s*end\s*=\s*([0-9.]+)\s*$", str(start_val), flags=re.I)
            if m:
                params["range"] = f"{m.group(1)}:{m.group(2)}"
    return params


def parse_time_range(params: Dict[str, str], default_begin: float, default_end: float) -> tuple:
    range_text = params.get("range") or params.get("time") or params.get("segment")
    if not range_text and params.get("_positional"):
        for item in params["_positional"]:
            if ":" in item:
                range_text = item
                break
    if not range_text or ":" not in range_text:
        return default_begin, default_end
    try:
        begin_s, end_s = range_text.split(":", 1)
        return float(begin_s), float(end_s)
    except ValueError:
        return default_begin, default_end


def safe_read_with_lock(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return pickle.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f'读取文件时出错 {file_path}: {e}')
        return None


def list_to_sha256(lst):
    json_str = json.dumps(lst, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _jsonl_path_for_save(save_path: str) -> str:
    if ".json.part" in save_path:
        return save_path.replace(".json.part", ".jsonl.part")
    if save_path.endswith(".json"):
        return save_path[:-5] + ".jsonl"
    return save_path + ".jsonl"


def _query_gpu_free_memory_mb() -> List[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return []
    mems = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            mems.append(int(line))
        except ValueError:
            continue
    return mems


def _configure_cuda_visible_devices(args) -> Optional[int]:
    """Set CUDA_VISIBLE_DEVICES for auto/manual selection.

    Returns the chosen physical GPU id if set, otherwise None.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None

    if getattr(args, "gpu_id", -1) >= 0:
        chosen = int(args.gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
        return chosen

    if not getattr(args, "auto_select_gpu", 0):
        return None

    mems = _query_gpu_free_memory_mb()
    if not mems:
        return None
    chosen = max(range(len(mems)), key=lambda i: mems[i])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    return chosen


# =============================================================================
# 主处理类
# =============================================================================

class SingleSampleProcessor:
    
    def __init__(self, args):
        logger.info("🔧 Initializing SingleSampleProcessor...")
        self.args = args
        self._last_text2text_error: Optional[str] = None
        self.physical_gpu_id = _configure_cuda_visible_devices(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
        
        # API 配置
        def _split_env(name, default=""):
            value = os.getenv(name, default)
            return [item for item in value.split(",") if item]

        self.ds_model_name = os.getenv('API_MODEL_NAME')
        self.ds_api_base = _split_env('API_BASE_URL')
        self.ds_api_keys = _split_env('API_KEY')
        if not self.ds_api_base:
            self.ds_api_base = _split_env('OPENAI_BASE_URL')
        if not self.ds_api_keys:
            self.ds_api_keys = _split_env('OPENAI_API_KEY')

        self.temporal_grounding_model_name = os.getenv(
            'API_MODEL_NAME_TEMPORAL_GROUNDING',
            self.ds_model_name or 'gpt-5.1'
        )
        self.temporal_grounding_api_base = _split_env(
            'API_BASE_URL_TEMPORAL_GROUNDING',
            ",".join(self.ds_api_base) if self.ds_api_base else ""
        )
        self.temporal_grounding_api_keys = _split_env('API_KEY_TEMPORAL_GROUNDING')
        if not self.temporal_grounding_api_base:
            self.temporal_grounding_api_base = self.ds_api_base
        if not self.temporal_grounding_api_keys:
            self.temporal_grounding_api_keys = self.ds_api_keys
        # ASR (OpenAI Whisper) 配置：仅用于 Defeasible Entailment（可通过环境变量关闭）
        self.asr_enable_defeasible = os.getenv("DEFEASIBLE_ASR_ENABLE", "1") == "1"
        self.asr_model_name = os.getenv("API_MODEL_NAME_ASR", "whisper-1")
        self.asr_api_base = os.getenv("API_BASE_URL_ASR", "https://api.openai.com/v1").strip()
        self.asr_api_key = (
            os.getenv("API_KEY_ASR", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
            or (self.ds_api_keys[0] if self.ds_api_keys else "")
        )
        self.asr_cache_dir = os.getenv(
            "ASR_CACHE_DIR",
            "./model_cache/asr",
        ).strip()
        os.makedirs(self.asr_cache_dir, exist_ok=True)
        self.vlm_model_name = os.getenv('API_MODEL_NAME_VLM')
        # VLM 远端（vLLM OpenAI 兼容服务）配置：仅当显式开启时使用
        vlm_remote_url_raw = os.getenv("API_BASE_URL_VLM", "").strip()
        self.vlm_use_remote = (os.getenv("VLM_USE_REMOTE", "0") == "1") or bool(vlm_remote_url_raw)
        self.vlm_api_base = _split_env("API_BASE_URL_VLM")
        self.vlm_api_keys = _split_env("API_KEY_VLM")
        if self.vlm_use_remote:
            if not self.vlm_api_base:
                raise ValueError("VLM_USE_REMOTE=1 但未设置 API_BASE_URL_VLM（示例: http://<ip>:8000/v1）")
            if not self.vlm_api_keys:
                # vLLM 常见用法是 API_KEY=EMPTY；若未给 VLM 专用 key，则沿用通用 key
                self.vlm_api_keys = self.ds_api_keys
        
        # 初始化 VLM
        self._init_vlm()
        
        # 初始化 Processor
        if not self.vlm_use_remote:
            self.processor = AutoProcessor.from_pretrained(
                self.vlm_model_name, 
                use_fast=True,
                trust_remote_code=True
            )
            self._set_processor_padding()
        else:
            self.processor = None
        
        # 检测模型类型
        model_path = (self.vlm_model_name or "").lower()
        forced_type = os.getenv("VLM_MODEL_TYPE", "").strip().lower()
        if forced_type in {"qwen", "internvl"}:
            self.model_type = forced_type
        else:
            self.model_type = 'internvl' if 'internvl' in model_path else 'qwen'
        logger.info(f"🔍 模型类型: {self.model_type}")
        
        # 路径配置
        self.full_dataset_folder = f'{args.dataset_folder}/{args.dataset}'
        self.clip_save_folder = f'{self.full_dataset_folder}/clips/{args.clip_duration}/'
        
        # 初始化检索器
        self.retriever = Retrieval_Manager(args, clip_save_folder=self.clip_save_folder)
        gpu_id = args.thread_idx % torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.retriever.load_model_to_gpu(gpu_id)
        
        # 问题去重相关
        self.STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at",
            "for", "and", "or", "this", "that", "these", "those", "it", "video", "clip",
            "please", "describe", "what", "happening", "happens", "happen", "doing", "do", "does"
        }
        self.TEMPLATE_PATTERNS = [
            r"\bdescribe\b.*\bvideo\b",
            r"\bwhat\b.*\bhappening\b",
            r"\bwhat\b.*\bmain action\b",
        ]
        self.FALLBACK_QUESTIONS = [
            "What is the main action in this video?",
            "What objects are involved and how do they move?",
            "What is the key event that happens near the end?",
            "What changes from start to finish?",
            "Who/what is the focus and what do they do?"
        ]
        
        self.experience_pool = []
        skills_dirs = None
        if getattr(args, "skills_dirs", ""):
            skills_dirs = [p for p in args.skills_dirs.split(os.pathsep) if p]
        else:
            prompt_ver = str(getattr(args, "prompt_version", "default")).lower()
            if prompt_ver == "short":
                # short 模式默认只加载轻量 skill，降低 system prompt 体积
                skills_dirs = [".codex/skills/short"]
            else:
                # default/minimal 模式：默认加载除 short 外的全部 skills
                # 避免 short 策略污染非 short 的工具选择行为。
                skills_root = Path(__file__).resolve().parents[1] / ".codex" / "skills"
                if skills_root.exists():
                    selected = [
                        str(p) for p in sorted(skills_root.iterdir())
                        if p.is_dir() and p.name.lower() != "short"
                    ]
                    if selected:
                        skills_dirs = selected
        if getattr(args, "enable_skills", 1):
            self.skills_text = format_skills_for_prompt(skills_dirs)
        else:
            self.skills_text = ""

        # Per-sample ephemeral cache for pretool_sweep to reuse extracted frames.
        # Key: (video_path, begin, end, fps) -> (frame_paths, timestamps)
        self._pretool_frame_cache: Dict[tuple, tuple] = {}

    def _init_vlm(self):
        """初始化 vLLM 服务"""
        logger.info(f"API_MODEL_NAME_VLM={self.vlm_model_name}")
        if self.vlm_use_remote:
            chosen_base = self.vlm_api_base[0]
            chosen_key = self.vlm_api_keys[0] if self.vlm_api_keys else ""
            self.vlm_client = OpenAI(base_url=chosen_base, api_key=chosen_key)
            self.vlm_server = None
            logger.info(f"✅ 使用远端 VLM（OpenAI兼容）: base_url={chosen_base}, model={self.vlm_model_name}")
            return
        
        model_path = self.vlm_model_name.lower()
        gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.70"))
        if 'internvl' in model_path:
            limit_mm = {"image": 20}
            mm_max_len = 16384
        else:
            limit_mm = {"video": 1}
            mm_max_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "24576"))
        
        try:
            self.vlm_server = LLM(
                model=self.vlm_model_name,
                gpu_memory_utilization=gpu_mem_util,
                tensor_parallel_size=1,
                max_model_len=mm_max_len,
                trust_remote_code=True,
                enforce_eager=True,
                enable_chunked_prefill=False,
                enable_prefix_caching=False,
                disable_mm_preprocessor_cache=True,
                limit_mm_per_prompt=limit_mm,
            )
            logger.info(f"✅ vLLM init OK (limit_mm={limit_mm}, max_len={mm_max_len})")
        except Exception as e:
            logger.error(f"❌ vLLM init failed: {e}")
            raise

    def _set_processor_padding(self):
        """设置 processor 的 padding_side"""
        try:
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'padding_side'):
                self.processor.tokenizer.padding_side = 'left'
            elif hasattr(self.processor, 'padding_side'):
                self.processor.padding_side = 'left'
        except (AttributeError, TypeError):
            pass

    def _get_video_ids(self, dic: Dict) -> List[str]:
        return list(dic.get("video_ids") or VIDEO_IDS)

    def _resolve_video_id(self, vid_id: str, dic: Dict) -> str:
        """将工具里出现的文件名/路径/带前缀的ID映射回真实 video_id。"""
        if not vid_id:
            return vid_id
        video_map = dic.get("video_map", {}) or {}
        if vid_id in video_map:
            return vid_id

        raw = str(vid_id).strip()
        raw = re.sub(r"^(video|vid)\s*", "", raw, flags=re.I).strip()
        raw = raw.strip("\"'<>:").strip()
        if raw in video_map:
            return raw

        raw_base = os.path.basename(raw)
        for k, path in video_map.items():
            if not path:
                continue
            base = os.path.basename(str(path))
            if raw == base or raw_base == base:
                return k

        raw_noext = os.path.splitext(raw_base)[0]
        if raw_noext:
            for k, path in video_map.items():
                base = os.path.basename(str(path))
                base_noext = os.path.splitext(base)[0]
                if raw_noext == base_noext:
                    return k
        return vid_id

    def _get_answer_choices(self, dic: Dict) -> List[str]:
        choices = dic.get("answer_choices")
        if choices:
            cleaned = [str(c).strip() for c in choices if str(c).strip()]
            # Preserve case for word/phrase choices (e.g. Strengthener/Weakener).
            # Uppercase only for single-letter choices.
            if cleaned and all(len(c) == 1 for c in cleaned):
                return [c.upper() for c in cleaned]
            return cleaned
        return list(ANSWER_LIST)

    def _invalid_answer_prompt(self, choices: List[str]) -> str:
        if choices and all(isinstance(c, str) and len(c.strip()) == 1 for c in choices):
            return f"Invalid! Choose one of {', '.join(choices)}. Reply with ONLY <answer>X</answer>."
        return f"Invalid! Choose EXACTLY one of: {', '.join(choices)}. Output ONLY that word/phrase."

    # =========================================================================
    # 字幕处理
    # =========================================================================

    def _load_subtitles_for_video(self, video_path: str) -> str:
        """Load subtitles for a single video file, return subtitle string."""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = f'{self.full_dataset_folder}/subtitles/{video_id}.srt'
        json_path = f'{self.full_dataset_folder}/subtitles/{video_id}.json'

        if os.path.exists(srt_path):
            subtitles = ''
            with open(srt_path, "r", encoding="utf-8") as file:
                content = file.read().split("\n\n")
                for section in content:
                    if section.strip():
                        lines = section.split("\n")
                        if len(lines) >= 3:
                            time_range = lines[1].split(" --> ")
                            start_time = parse_subtitle_time(time_range[0])
                            end_time = parse_subtitle_time(time_range[1])
                            text = " ".join(lines[2:])
                            pattern = r'<font color="white" size=".72c">(.*?)</font>'
                            raw_text = re.findall(pattern, text, flags=re.DOTALL)
                            text = raw_text[0] if raw_text else text
                            subtitles += f'{int(start_time)}-{int(end_time)}:{text} '
            return subtitles
        elif os.path.exists(json_path):
            subtitles = json.load(open(json_path))
            return '\n'.join([
                f'{int(parse_subtitle_time(sub["start"]))}-{int(parse_subtitle_time(sub["end"]))}:{sub["line"]}'
                for sub in subtitles
            ])
        return ''

    def get_dic_subtitles(self, dic):
        _task = dic.get("task", "") or getattr(self.args, "dataset", "")
        use_openai_asr = self.asr_enable_defeasible and is_defeasible_entailment_task(_task)
        video_map = dic.get('video_map', {})
        if video_map:
            subtitles_map = {}
            combined_parts = []
            for vid_id, path in video_map.items():
                if use_openai_asr:
                    sub = transcribe_video_openai_whisper_cached(
                        path,
                        api_base=self.asr_api_base,
                        api_key=self.asr_api_key,
                        model_name=self.asr_model_name,
                        cache_dir=self.asr_cache_dir,
                        task_name="Defeasible Entailment",
                    )
                    if not sub:
                        sub = self._load_subtitles_for_video(path)
                else:
                    sub = self._load_subtitles_for_video(path)
                subtitles_map[vid_id] = sub
                if sub:
                    combined_parts.append(f"[Video {vid_id}] {sub}")
            dic['subtitles_map'] = subtitles_map
            dic['subtitles'] = '\n'.join(combined_parts) if combined_parts else ''
        else:
            single_video = dic.get('video_path', '')
            if use_openai_asr:
                sub = (
                    transcribe_video_openai_whisper_cached(
                        single_video,
                        api_base=self.asr_api_base,
                        api_key=self.asr_api_key,
                        model_name=self.asr_model_name,
                        cache_dir=self.asr_cache_dir,
                        task_name="Defeasible Entailment",
                    )
                    if single_video else ""
                )
                dic['subtitles'] = sub or self._load_subtitles_for_video(single_video)
            else:
                dic['subtitles'] = self._load_subtitles_for_video(single_video)
        return dic

    # =========================================================================
    # 问题去重
    # =========================================================================

    def normalize_question(self, q: str) -> str:
        if not q:
            return ""
        q = q.lower().strip()
        q = re.sub(r"<[^>]+>", " ", q)
        q = re.sub(r"[^a-z0-9\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        for pat in self.TEMPLATE_PATTERNS:
            if re.search(pat, q):
                return "__TEMPLATE_QUESTION__"
        return q

    def token_set(self, q: str):
        nq = self.normalize_question(q)
        if not nq:
            return set()
        return {t for t in nq.split() if t not in self.STOPWORDS and len(t) > 2}

    def jaccard(self, a: str, b: str) -> float:
        A, B = self.token_set(a), self.token_set(b)
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def too_similar(self, q: str, history: List[str], *, 
                   exact_norm: bool = True, seq_threshold: float = 0.95,
                   jacc_threshold: float = 0.85, min_len_for_seq: int = 20,
                   max_history: int = 5) -> bool:
        """检查问题是否与历史过于相似（参数已优化）"""
        if not q:
            return True
        nq = self.normalize_question(q)
        if not nq or nq == "__TEMPLATE_QUESTION__":
            return True
        if len(nq) < 30:
            return False

        recent = history[-max_history:]
        for old in recent:
            no = self.normalize_question(old)
            if not no:
                continue
            if exact_norm and nq == no:
                return True
            if len(nq) >= min_len_for_seq and len(no) >= min_len_for_seq:
                if SequenceMatcher(None, nq, no).ratio() >= seq_threshold:
                    return True
            if self.jaccard(q, old) >= jacc_threshold:
                return True
        return False

    # =========================================================================
    # Prompt 构建
    # =========================================================================

    def _sanitize_question(self, text: str, min_len: int = 5, fallback: str = "") -> str:
        if not text:
            return fallback
        text = re.sub(r'<[^>]+>', '', text).strip().strip('"\'')
        return text if len(text) >= min_len else fallback

    def _summarize_descriptions(self, descriptions: Dict[str, str], max_len: int = 250, video_ids: Optional[List[str]] = None) -> List[str]:
        ids = video_ids or list(VIDEO_IDS)
        return [f"**{vid}**: {descriptions.get(vid, '[N/A]')[:max_len]}" for vid in ids]

    def _synthesize_per_video_summary(
        self,
        vid_id: str,
        all_tool_results: Dict[str, str],
        tool_results: Dict[str, List[str]],
    ) -> str:
        """把同一视频的所有工具结果压成一句简洁结论（用于最终决策）。

        设计原则：
        - 逐帧原始数据已在之前的 cross-check 中展示过，summary 不再重复。
        - Summary = VLM 结论 + detection 工具的 **高层概括** + 冲突修正标记。
        - 目标：让 LLM 在做最终决策时一眼看清每个视频的关键属性。
        """
        vlm_text = all_tool_results.get(vid_id, "").strip()

        # ── VLM 结论（只保留最关键一两句，避免与原证据重复）──
        vlm_summary = ""
        if vlm_text:
            cap, ev = _split_caption_evidence(vlm_text)
            def _first_sentence(s: str) -> str:
                s = s.strip()
                if not s:
                    return ""
                parts = re.split(r"[.。?!]\s+", s, maxsplit=1)
                return parts[0].strip()
            cap_1 = _first_sentence(cap)
            ev_1 = _first_sentence(ev)
            snippets = []
            if cap_1:
                snippets.append(f"cap: {cap_1}")
            if ev_1:
                snippets.append(f"ev: {ev_1}")
            vlm_summary = "; ".join(snippets)[:220]

        # ── Detection 工具高层概括（只保留关键属性，不 dump 逐帧数据）──
        det_highlights: List[str] = []
        if tool_results:
            # 合并 scene_graph + object_tracker 的原始文本
            all_det_text: List[str] = []
            for tool_name in ("scene_graph", "object_tracker"):
                for block in tool_results.get(tool_name, []):
                    extracted = self._extract_vid_section_from_block(block, vid_id)
                    if extracted:
                        all_det_text.append(extracted)
            if all_det_text:
                combined = "\n".join(all_det_text)
                sig_color, nf, tot, nobj = _det_signal_color_profile(combined)
                # 只在置信度足够时展示（≥30% 帧数且 ≥2 帧），避免噪声误导
                if sig_color and tot > 0 and nf >= 2 and nf / tot >= 0.30:
                    det_highlights.append(f"signal={sig_color} ({nf}/{tot} frames)")

        # ── 组装 ──
        parts: List[str] = []
        if vlm_summary:
            parts.append(vlm_summary)
        if det_highlights:
            parts.append("det: " + "; ".join(det_highlights))

        if not parts:
            return "[N/A]"
        return " | ".join(parts)

    def _summarize_tool_results(
        self,
        all_tool_results: Dict[str, str],
        video_ids: Optional[List[str]] = None,
        tool_results: Optional[Dict[str, List[str]]] = None,
    ) -> List[str]:
        """生成 per-video 压缩摘要列表。如果有确定性工具数据，走方案B聚合；否则 fallback。"""
        ids = video_ids or list(VIDEO_IDS)
        if tool_results:
            return [
                f"- {vid}: {self._synthesize_per_video_summary(vid, all_tool_results, tool_results)}"
                for vid in ids
            ]
        return [f"- {vid}: {all_tool_results.get(vid, '[N/A]')[:800]}" for vid in ids]

    def _llm_progress_summary_short(
        self,
        dic: Dict[str, Any],
        messages: List[Dict[str, Any]],
        video_ids: List[str],
        all_tool_results: Dict[str, str],
    ) -> str:
        """Use LLM to distill recent progress for short mode compression."""
        prompt_version = str(getattr(self.args, "prompt_version", "default") or "default").lower()
        if prompt_version == "short":
            pass
        else:
            return ""

        recent = messages[-6:] if len(messages) > 6 else messages
        recent_text = []
        for m in recent:
            role = str(m.get("role", "")).upper()
            content = self._flatten_message_text(m.get("content", ""))
            content = self._strip_think_content(content)[:1200]
            if content:
                recent_text.append(f"{role}: {content}")

        done = [vid for vid in video_ids if all_tool_results.get(vid)]
        remain = [vid for vid in video_ids if vid not in done]
        prompt = (
            "You are compressing context for a video QA agent.\n"
            "Summarize progress into <= 6 lines, plain text only.\n"
            "Must include:\n"
            "1) What has been verified\n"
            "2) Which videos are still missing/uncertain\n"
            "3) Exactly one best next action\n"
            "No chain-of-thought.\n\n"
            f"Question: {dic.get('question', '')}\n"
            f"Done videos: {done}\n"
            f"Remaining videos: {remain}\n\n"
            "Recent dialogue:\n"
            + "\n".join(recent_text)
        )
        sum_messages = [{"role": "user", "content": prompt}]

        old_err = self._last_text2text_error
        try:
            summary = self.single_text2text(sum_messages, self.ds_model_name, self.ds_api_base, self.ds_api_keys)
        finally:
            # Keep main loop error state intact.
            self._last_text2text_error = old_err
        summary = str(summary or "")
        # 防止摘要里混入 think/answer/tag，污染后续轮次决策
        summary = self._strip_think_content(summary)
        summary = re.sub(r"<answer>.*?</answer>", "", summary, flags=re.I | re.S)
        summary = re.sub(r"</?answer>", "", summary, flags=re.I)
        summary = re.sub(r"<\s*/?\s*think\s*>", "", summary, flags=re.I)
        return summary.strip()

    def _maybe_compress_context_short(
        self,
        messages: List[Dict[str, Any]],
        dic: Dict[str, Any],
        all_tool_results: Dict[str, str],
        tool_results: Dict[str, List[str]],
        video_ids: List[str],
        cur_turn: int,
    ) -> bool:
        """In short mode, proactively compact context before long calls."""
        prompt_version = str(getattr(self.args, "prompt_version", "default") or "default").lower()
        if prompt_version == "short":
            if len(messages) <= 4:
                return False
        else:
            return False

        min_messages = int(os.getenv("SHORT_CTX_COMPRESS_MIN_MESSAGES", "8"))
        token_threshold = int(os.getenv("SHORT_CTX_COMPRESS_TOKEN_THRESHOLD", "10000"))
        estimated_tokens = self._estimate_chat_input_tokens(messages)
        if len(messages) < min_messages and estimated_tokens < token_threshold:
            return False

        old_len = len(messages)
        new_messages: List[Dict[str, Any]] = []

        # Keep system + initial user instruction.
        idx = 0
        if messages and messages[0].get("role") == "system":
            new_messages.append(messages[0])
            idx = 1
        for m in messages[idx:]:
            if m.get("role") == "user":
                new_messages.append(m)
                break

        # Add compact evidence summary.
        summary_lines = self._summarize_tool_results(
            all_tool_results,
            video_ids=video_ids,
            tool_results=tool_results,
        )
        remain = [vid for vid in video_ids if vid not in all_tool_results]
        compressed_msg = {
            "role": "user",
            "content": (
                f"[System] Context compressed at turn {cur_turn}.\n"
                f"Question: {dic.get('question', '')}\n"
                "Evidence summary:\n"
                + "\n".join(summary_lines)
                + f"\nRemaining videos: {remain if remain else '[]'}\n"
                "Next: call one needed tool or output <answer>X</answer>."
            ),
        }
        new_messages.append(compressed_msg)

        # Replace raw carry-over with model-distilled progress summary.
        use_llm_summary = os.getenv("SHORT_CTX_USE_LLM_SUMMARY", "1") == "1"
        if use_llm_summary:
            llm_summary = self._llm_progress_summary_short(
                dic=dic,
                messages=messages,
                video_ids=video_ids,
                all_tool_results=all_tool_results,
            )
            if llm_summary:
                new_messages.append(
                    {
                        "role": "user",
                        "content": "[System] Progress Summary:\n" + llm_summary,
                    }
                )
        if len(new_messages) <= 2:
            # Fallback: keep minimal recency when summary is unavailable.
            for m in messages[-2:]:
                if not any(m == x for x in new_messages):
                    new_messages.append(m)

        messages[:] = new_messages
        logger.info(
            f"[SHORT_CTX_COMPRESS] turn={cur_turn}, msgs {old_len}->{len(messages)}, "
            f"est_tokens={estimated_tokens}, threshold={token_threshold}"
        )
        return True

    @staticmethod
    def _camera_hint_from_path(path: str) -> str:
        if not path:
            return ""
        name = os.path.basename(str(path))
        if "CAM_" not in name:
            return ""
        cam = name.split("CAM_", 1)[1].split(".", 1)[0]
        return cam.replace("_", " ").lower().strip()

    def _focus_video_ids_by_question(self, dic: Dict, video_ids: List[str]) -> List[str]:
        """If question targets specific views (front/left/right), focus summary on those."""
        question = (dic.get("question") or "").lower()
        if not question:
            return video_ids

        wanted: List[str] = []
        if re.search(r'\b(left front|front left)\b', question):
            wanted.append("front left")
        if re.search(r'\b(right front|front right)\b', question):
            wanted.append("front right")
        # check "front" after removing left/right front
        q2 = re.sub(r'\b(left front|front left|right front|front right)\b', ' ', question)
        if re.search(r'\bfront\b', q2):
            wanted.append("front")

        if not wanted:
            return video_ids

        cam_map: Dict[str, str] = {}
        for vid in video_ids:
            path = (dic.get("video_map") or {}).get(vid, "")
            cam_map[vid] = self._camera_hint_from_path(path)

        focused: List[str] = []
        for cam in wanted:
            for vid, hint in cam_map.items():
                if hint == cam and vid not in focused:
                    focused.append(vid)
        return focused or video_ids

    def _is_similarity_centric_question(self, dic: Dict) -> bool:
        text = ((dic.get("question") or "") + " " + " ".join(dic.get("options", []) or [])).lower()
        return bool(re.search(r"\b(similar|similarity|most similar|style|look|appearance)\b", text))

    def _similarity_rank_signal(
        self,
        dic: Dict,
        required_videos: set,
        tool_results: Optional[Dict[str, List[str]]],
    ) -> tuple:
        """Parse visual_similarity outputs and return readiness signal.

        Returns:
            (ready, margin, top_vid, score_map)
        """
        if not tool_results:
            return False, 0.0, None, {}
        blocks = tool_results.get("visual_similarity", []) or []
        if not blocks:
            return False, 0.0, None, {}

        pair_re = re.compile(r"([A-Za-z0-9_.\-]+)↔([A-Za-z0-9_.\-]+)=([0-9]*\.?[0-9]+)")
        score_map: Dict[str, float] = {}
        for blk in blocks:
            for a_raw, b_raw, s_raw in pair_re.findall(blk or ""):
                try:
                    score = float(s_raw)
                except Exception:
                    continue
                a = self._resolve_video_id(a_raw, dic)
                b = self._resolve_video_id(b_raw, dic)
                # Prefer ref↔candidate pairs; fallback to any candidate pair involving required videos.
                if a == "ref" and b in required_videos:
                    score_map[b] = max(score_map.get(b, -1e9), score)
                elif b == "ref" and a in required_videos:
                    score_map[a] = max(score_map.get(a, -1e9), score)

        candidates = sorted([v for v in required_videos if v != "ref"])
        if not candidates:
            return False, 0.0, None, score_map
        if any(v not in score_map for v in candidates):
            return False, 0.0, None, score_map

        ranked = sorted([(v, score_map[v]) for v in candidates], key=lambda x: x[1], reverse=True)
        top_vid, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = float(top_score - second_score)
        # Similarity-first policy: once ref↔all-candidates scores are available,
        # allow answering without extra video_reader gating.
        ready = True
        return ready, margin, top_vid, score_map

    def _format_final_summary(self, descriptions: Dict[str, str], max_len: int = 200, video_ids: Optional[List[str]] = None) -> str:
        ids = video_ids or list(VIDEO_IDS)
        return "\n".join([f"- {vid}: {descriptions.get(vid, '[N/A]')[:max_len]}" for vid in ids])

    def _force_answer(self, messages: List[Dict[str, Any]], choices: List[str]) -> str:
        prompt = f"You MUST choose one of {', '.join(choices)} NOW. Output: <answer>X</answer>"
        self._append_message(messages, "user", prompt)
        retry = self.single_text2text(messages, self.ds_model_name, self.ds_api_base, self.ds_api_keys)
        self._append_message(messages, "assistant", retry)
        return retry

    def _append_message(self, messages: List[Dict[str, Any]], role: str, content: Any) -> None:
        # 仅在 short prompt 下剥离 <think>，减少上下文膨胀和时延
        prompt_version = str(getattr(self.args, "prompt_version", "default") or "default").lower()
        if prompt_version == "short":
            if role == "assistant" and isinstance(content, str):
                content = self._strip_think_content(content)
        else:
            pass
        messages.append({"role": role, "content": content})

    def _count_tool_calls_in_text(self, text: str) -> int:
        """Count executable tool tags in one assistant turn."""
        if not isinstance(text, str) or not text.strip():
            return 0
        return (
            len(RE_VIDEO_READER.findall(text))
            + len(RE_SCENE_DETECTOR.findall(text))
            + len(RE_OBJECT_TRACKER.findall(text))
            + len(RE_SPATIAL_RELATION.findall(text))
            + len(RE_SCENE_GRAPH.findall(text))
            + len(RE_VISUAL_SIMILARITY.findall(text))
            + len(RE_TG_AGENT.findall(text))
            + len(RE_SEGMENT_QUERY.findall(text))
            + len(RE_SUBTITLE_RETRIEVER.findall(text))
            + len(RE_SUBTITLE_EXTRACTOR.findall(text))
            + len(RE_VIDEO_BROWSER.findall(text))
        )

    def _strip_think_content(self, text: str) -> str:
        """Remove think blocks, including unclosed '<think>' tails."""
        if not isinstance(text, str) or not text:
            return ""
        # First drop normal closed blocks.
        cleaned = RE_TAG_THINK.sub("", text)
        # Then drop unclosed think tails: '<think> ...EOF'
        cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.I)
        return cleaned.strip()

    def _append_tool_result(self, tool_results: Optional[Dict[str, List[str]]], tool_name: str, result: str) -> None:
        if tool_results is None or not result:
            return
        tool_results.setdefault(tool_name, []).append(result.strip())

    def _extract_prompt_keywords(self, text: str, max_items: int = 6) -> List[str]:
        """Extract a small set of generic visual keywords from question/options.
        Keep this dataset-agnostic and conservative.
        """
        if not text:
            return []
        t = text.lower()
        vocab = [
            "person", "man", "woman", "child",
            "hand", "finger", "face",
            "car", "truck", "bus", "bicycle", "motorcycle",
            "traffic light", "sign",
            "cat", "dog", "bird",
            "ball", "basketball",
            "hat", "dress", "belt",
            "robot", "helicopter",
        ]
        hits = []
        for kw in vocab:
            if kw in t:
                hits.append(kw)
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for h in hits:
            if h not in seen:
                uniq.append(h)
                seen.add(h)
        return uniq[:max_items]

    def _build_pretool_sweep_calls(self, dic: Dict, *, video_ids: List[str], answer_choices: List[str]) -> str:
        """Build a synthetic tool-call text to pre-run tools before LLM planning.
        This is optional and can be expensive; keep defaults conservative.
        """
        q_text = dic.get("question", "")
        if dic.get("options"):
            q_text = q_text + "\n" + "\n".join(dic.get("options") or [])
        keywords = self._extract_prompt_keywords(q_text, max_items=6)
        prompts = ",".join(keywords) if keywords else "person,car"
        # pick 1-2 for relation
        rel_targets = ",".join(keywords[:2]) if len(keywords) >= 2 else ("person,car" if not keywords else f"{keywords[0]},car")
        obj_target = keywords[0] if keywords else "person"

        max_secs = float(getattr(self.args, "pretool_max_seconds", 10.0))
        sweep_fps = float(getattr(self.args, "pretool_fps", 2.0))

        # Use bounded range to control cost.
        # ≥5 视频时只跑 scene_graph（砍掉 scene_detector / object_tracker / spatial_relation，
        # 避免 context 膨胀到 ~3000字 导致 LLM 空响应）。
        many_videos = len(video_ids) >= 5
        lines = []
        for vid in video_ids:
            dur = float(dic.get("duration_map", {}).get(vid, 10.0))
            end = min(max_secs, max(1.0, dur))
            # NOTE: pre-tool sweep 默认不跑 video_reader，以避免 VLM 幻觉污染上下文
            if not many_videos:
                # scene changes
                lines.append(f'<scene_detector id="{vid}">fps=1;threshold=0.6</scene_detector>')
                # generic detection / relations
                lines.append(f'<object_tracker id="{vid}">0:{end:.1f};target={obj_target};fps={sweep_fps};conf=0.25</object_tracker>')
                lines.append(f'<spatial_relation id="{vid}">0:{end:.1f};targets={rel_targets};fps={sweep_fps};conf=0.25</spatial_relation>')
            # scene graph — always run (most informative single tool)
            lines.append(f'<scene_graph id="{vid}">0:{end:.1f};prompts={prompts};model=groundingdino</scene_graph>')

            # subtitles only when enabled
            if getattr(self.args, "use_subtitle", 0):
                lines.append(f'<subtitle_retriever id="{vid}">{dic.get("question","")}</subtitle_retriever>')
                lines.append(f'<subtitle_extractor id="{vid}">0:{end:.1f}</subtitle_extractor>')

        # visual similarity: run for all tasks (use ref if exists; otherwise use the first video as baseline)
        baseline = None
        if "ref" in dic.get("video_map", {}) and "ref" in video_ids:
            baseline = "ref"
        elif video_ids:
            baseline = video_ids[0]
        if baseline:
            dur_base = float(dic.get("duration_map", {}).get(baseline, 10.0))
            end_base = min(max_secs, max(1.0, dur_base))
            for cand in video_ids:
                if cand == baseline:
                    continue
                dur_c = float(dic.get("duration_map", {}).get(cand, 10.0))
                end_c = min(max_secs, max(1.0, dur_c))
                lines.append(
                    f'<visual_similarity id="{baseline},{cand}">a=0:{end_base:.1f};b=0:{end_c:.1f};fps={sweep_fps};model=clip</visual_similarity>'
                )

        return "\n".join(lines)

    # ICL 任务是纯推理（从 (video, answer) 示例推断规律），不需要 detection 工具证据。
    # pretool_sweep 只会膨胀 context 导致 API 空响应。
    _PRETOOL_SKIP_TASKS = frozenset({"ICL"})

    def _run_pretool_sweep(
        self,
        dic: Dict,
        *,
        messages: List[Dict[str, Any]],
        tool_cache: Dict[str, str],
        tool_results: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Optionally run a pre-tool sweep and append results into context."""
        if not getattr(self.args, "pretool_sweep", 0):
            return
        # ICL 等纯推理任务跳过 pretool_sweep，避免 context 膨胀
        task = dic.get("task", "")
        if isinstance(task, list):
            task = task[0] if task else ""
        if task in self._PRETOOL_SKIP_TASKS:
            logger.info(f"[PRETOOL] Skipping pretool_sweep for task={task} (pure reasoning)")
            return
        # Reset per-sample cache
        self._pretool_frame_cache = {}
        video_ids = self._get_video_ids(dic)
        answer_choices = self._get_answer_choices(dic)

        # Pre-extract frames once per video for the sweep range, so vision tools can reuse.
        max_secs = float(getattr(self.args, "pretool_max_seconds", 10.0))
        sweep_fps = float(getattr(self.args, "pretool_fps", 2.0))
        for vid in video_ids:
            if vid not in dic.get("video_map", {}):
                continue
            video_path = dic["video_map"][vid]
            duration = float(dic.get("duration_map", {}).get(vid, DEFAULT_VIDEO_DURATION))
            end = float(f"{min(max_secs, max(1.0, duration)):.1f}")
            try:
                frames, timestamps = timestamp_to_clip_path(
                    self.full_dataset_folder, 0.0, float(end), video_path, fps=sweep_fps
                )
                if frames:
                    key = (video_path, 0.0, float(end), float(sweep_fps))
                    self._pretool_frame_cache[key] = (frames, timestamps)
            except Exception:
                pass
        sweep_text = self._build_pretool_sweep_calls(dic, video_ids=video_ids, answer_choices=answer_choices)
        tool_result = self.process_tool_calls(sweep_text, dic, tool_cache, tool_results)
        if tool_result:
            # Keep it as "evidence" for the planner; still allow re-check with video_reader.
            self._append_message(
                messages,
                "user",
                "Pre-tool evidence:\n" + tool_result,
            )

    def _handle_tool_result(
        self,
        tool_result: str,
        pending_videos: set,
        cur_turn: int,
        all_tool_results: Dict[str, str],
        messages: List[Dict[str, Any]],
        dic: Optional[Dict[str, Any]] = None,
        video_ids: Optional[List[str]] = None,
        tool_results: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        if not tool_result:
            return False
        if pending_videos and cur_turn < MAX_DS_ROUND - 3:
            tool_result += f"\n⚠️ Remaining: {sorted(pending_videos)}. Continue reading!"
        else:
            focus_ids = video_ids
            if dic is not None and video_ids:
                focus_ids = self._focus_video_ids_by_question(dic, video_ids)
            summary = self._summarize_tool_results(all_tool_results, video_ids=focus_ids, tool_results=tool_results)
            tool_result += "\n\n📊 **Summary:**\n" + "\n".join(summary)
            if dic is not None and focus_ids:
                uncertain_ids = self._find_uncertain_videos_in_tool_result(tool_result, focus_ids)
                if uncertain_ids and cur_turn < MAX_DS_ROUND - 1:
                    tool_result += (
                        "\n⚠️ Evidence unclear for videos: "
                        + ", ".join(uncertain_ids)
                        + ". You can:\n"
                        "- Re-read with a more specific question\n"
                        "- Use <scene_graph> to detect/count specific objects\n"
                        "- Use <visual_similarity> to compare two videos\n"
                        "Do NOT answer until the key attribute is confirmed."
                    )
                    # 证据不足时不追加 answer prompt，避免矛盾指令
                    self._append_message(messages, "user", tool_result)
                    return True
            answer_choices = []
            if dic is not None:
                answer_choices = dic.get("answer_choices") or []
            no_ref_version = getattr(self.args, "no_ref_prompt_version", "current")
            if dic is not None and dic.get("no_ref") and no_ref_version == "legacy":
                tool_result += "\nOutput your final answer inside <answer></answer> tags."
            elif answer_choices:
                if all(isinstance(c, str) and len(c.strip()) == 1 for c in answer_choices):
                    tool_result += "\nOutput your final answer (" + "/".join(answer_choices) + ") inside <answer></answer> tags."
                else:
                    tool_result += "\nOutput ONLY one of: " + " / ".join(answer_choices) + "."
            else:
                tool_result += "\nOutput your final answer inside <answer></answer> tags."
        self._append_message(messages, "user", tool_result)
        return True

    def _select_prompt_template(self, prompt_version: str):
        if prompt_version == 'minimal':
            return "minimal", initial_input_template_multivideo_minimal
        if prompt_version == 'short':
            template = (initial_input_template_multivideo_subtitle_short if self.args.use_subtitle
                        else initial_input_template_multivideo_wo_subtitle_short)
            return "short", template
        template = (initial_input_template_multivideo_subtitle if self.args.use_subtitle
                    else initial_input_template_multivideo_wo_subtitle)
        return "default", template

    @staticmethod
    def _detect_question_type_hints(question: str, options: List[str]) -> str:
        """Deprecated — guidance now embedded in prompt templates."""
        return ""

    def build_initial_prompt(self, data: Dict) -> str:
        data['question'] = data['question'].replace('Only give the best option. Best option: (', '')
        data['options'] = data.get('options', [])
        
        video_info_str = ""
        video_text_map = data.get('video_text_map', {})
        if 'video_map' in data:
            video_ids = data.get("video_ids") or list(data["video_map"].keys())
            for vid_id in video_ids:
                if vid_id not in data["video_map"]:
                    continue
                dur = data['duration_map'].get(vid_id, 0)
                # Only use basename to avoid leaking path/category/GT info to the LLM.
                file_name = os.path.basename(str(data["video_map"][vid_id]))
                camera_hint = ""
                if "CAM_" in file_name:
                    cam_part = file_name.split("CAM_", 1)[1].split(".", 1)[0]
                    cam_part = cam_part.replace("_", " ").lower()
                    camera_hint = f"; camera: {cam_part}"
                file_suffix = f" (file: {file_name}{camera_hint})" if file_name else ""
                video_info_str += f"- Video '{vid_id}': {dur:.1f}s{file_suffix}"
                if vid_id in video_text_map:
                    text_info = video_text_map[vid_id]
                    for key in ["cf_question", "cf_option1", "cf_option2"]:
                        if key in text_info:
                            video_info_str += f"\n  {key}: {text_info[key]}"
                video_info_str += "\n"
        
        question_text = data['question'] + "\n" + "\n".join(data['options'])
        if data.get("no_ref"):
            no_ref_version = getattr(self.args, "no_ref_prompt_version", "current")
            if no_ref_version == "legacy":
                template = initial_input_template_multivideo_no_ref_legacy
            else:
                template = initial_input_template_multivideo_no_ref
            base_prompt = template.format(
                question=question_text,
                video_info=video_info_str,
            )
        else:
            prompt_version = getattr(self.args, 'prompt_version', 'default')
            version, template = self._select_prompt_template(prompt_version)
            if version == "minimal":
                base_prompt = template.format(question=question_text, video_info=video_info_str)
            elif version == "short":
                base_prompt = template.format(question=question_text, video_info=video_info_str)
            else:
                base_prompt = template.format(question=question_text, video_info=video_info_str, MAX_DS_ROUND=MAX_DS_ROUND)
        
        # 追加基于问题类型的针对性提示
        type_hints = self._detect_question_type_hints(data['question'], data['options'])
        if type_hints:
            base_prompt += type_hints

        prompt = base_prompt.replace('thinking>', 'think>')
        return prompt

    # =========================================================================
    # API 调用
    # =========================================================================

    def _flatten_message_text(self, content: Any) -> str:
        """把 messages 中可见文本扁平化，用于粗略 token 预估。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(self._flatten_message_text(item) for item in content)
        if isinstance(content, dict):
            # OpenAI 多模态 content 常见结构：{"type":"text","text":"..."}
            if "text" in content and isinstance(content.get("text"), str):
                return content.get("text", "")
            # 兜底：把 dict 中字符串字段拼起来
            parts = []
            for v in content.values():
                if isinstance(v, (str, list, dict)):
                    parts.append(self._flatten_message_text(v))
            return " ".join(p for p in parts if p)
        return ""

    def _estimate_chat_input_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """粗略估算 chat 输入 token，避免明显超上下文的无效请求。"""
        if not messages:
            return 0
        total_chars = 0
        for m in messages:
            total_chars += len(str(m.get("role", "")))
            total_chars += len(self._flatten_message_text(m.get("content", "")))
        # 默认按 ~3 chars/token 估算（偏保守），并加每条 message 的结构开销
        chars_per_token = float(os.getenv("TEXT_TOKEN_EST_CHARS_PER_TOKEN", "3.0"))
        chars_per_token = max(chars_per_token, 1.0)
        estimated = int(total_chars / chars_per_token) + len(messages) * 10 + 128
        return max(estimated, 0)
    
    def single_text2text(self, message, model_name, base_url=None, api_keys=None) -> str:
        """调用文本 LLM API"""
        start_time = time.time()
        self._last_text2text_error = None
        try:
            if not base_url or not api_keys:
                logger.error(f"[TEXT2TEXT] Missing base_url/api_key. base_url={base_url}, api_keys_set={bool(api_keys)}")
                return ""
            # 确定性选择 endpoint：总是用第一个，避免 random.choice 随调用次数漂移
            chosen_base = base_url[0]
            chosen_key = api_keys[0]
            if any(d in chosen_base for d in ("cognitiveservices.azure.com", "openai.azure.com", "services.ai.azure.com")):
                api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
                llm = AzureOpenAI(
                    azure_endpoint=chosen_base,
                    api_key=chosen_key,
                    api_version=api_version,
                )
            else:
                llm = OpenAI(base_url=chosen_base, api_key=chosen_key)
            for retry in range(API_RETRY_COUNT):
                try:
                    requested_max_tokens = int(os.getenv("TEXT_MAX_COMPLETION_TOKENS", "16384"))
                    context_limit = int(os.getenv("TEXT_MAX_CONTEXT_TOKENS", "40960"))
                    min_completion = int(os.getenv("TEXT_MIN_COMPLETION_TOKENS", "128"))
                    text_temperature = float(os.getenv("TEXT_TEMPERATURE", "0.0"))
                    text_top_p = float(os.getenv("TEXT_TOP_P", "1.0"))
                    text_temperature = min(max(text_temperature, 0.0), 2.0)
                    text_top_p = min(max(text_top_p, 0.0), 1.0)
                    estimated_input_tokens = self._estimate_chat_input_tokens(message)
                    available_tokens = context_limit - estimated_input_tokens - 64  # 安全余量

                    # 预判：剩余空间明显不足时，直接交给上层走 direct VLM fallback
                    if available_tokens < min_completion:
                        self._last_text2text_error = "max_tokens_too_large"
                        logger.warning(
                            f"[TEXT2TEXT] Precheck insufficient context, skip API call: "
                            f"estimated_input={estimated_input_tokens}, context={context_limit}, "
                            f"available={available_tokens}, min_completion={min_completion}"
                        )
                        return ""

                    max_completion_tokens = min(requested_max_tokens, available_tokens)
                    if max_completion_tokens < requested_max_tokens:
                        logger.info(
                            f"[TEXT2TEXT] Precheck clamp max_completion_tokens: "
                            f"{requested_max_tokens} -> {max_completion_tokens} "
                            f"(estimated_input={estimated_input_tokens}, context={context_limit})"
                        )
                    completion = llm.chat.completions.create(
                        model=model_name,
                        messages=message,
                        temperature=text_temperature,
                        top_p=text_top_p,
                        seed=GLOBAL_SEED,
                        max_completion_tokens=max_completion_tokens,
                    )
                    ans = completion.choices[0].message.content.strip()
                    usage = getattr(completion, "usage", None)
                    if usage is not None:
                        p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                        c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                        t_tok = int(getattr(usage, "total_tokens", (p_tok + c_tok)) or (p_tok + c_tok))
                        logger.info(
                            f"[TEXT2TEXT_USAGE] prompt_tokens={p_tok}, "
                            f"completion_tokens={c_tok}, total_tokens={t_tok}"
                        )
                    # 记录 server fingerprint 用于验证可复现性
                    fp = getattr(completion, 'system_fingerprint', None)
                    logger.info(
                        f"[TEXT2TEXT] ✅ {time.time()-start_time:.2f}s, len={len(ans)}"
                        + (f", fingerprint={fp}" if fp else "")
                    )
                    return ans
                except Exception as e:
                    err_str = str(e)
                    # 当 max_tokens 超过上下文剩余空间时，解析可用 token 数并立即缩容重试
                    if ("max_tokens" in err_str or "max_completion_tokens" in err_str) and "too large" in err_str:
                        self._last_text2text_error = "max_tokens_too_large"
                        m = re.search(r'(\d+)\s*>\s*(\d+)\s*-\s*(\d+)', err_str)
                        if m:
                            context_len = int(m.group(2))
                            input_tokens = int(m.group(3))
                            available = context_len - input_tokens - 64  # 留 64 token 安全余量
                            if available > 0:
                                logger.warning(
                                    f"[TEXT2TEXT] max_tokens too large ({max_completion_tokens}), "
                                    f"retrying with available={available} "
                                    f"(context={context_len}, input={input_tokens})"
                                )
                                try:
                                    completion = llm.chat.completions.create(
                                        model=model_name,
                                        messages=message,
                                        temperature=text_temperature,
                                        top_p=text_top_p,
                                        seed=GLOBAL_SEED,
                                        max_completion_tokens=available,
                                    )
                                    ans = completion.choices[0].message.content.strip()
                                    usage = getattr(completion, "usage", None)
                                    if usage is not None:
                                        p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                                        c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                                        t_tok = int(getattr(usage, "total_tokens", (p_tok + c_tok)) or (p_tok + c_tok))
                                        logger.info(
                                            f"[TEXT2TEXT_USAGE] prompt_tokens={p_tok}, "
                                            f"completion_tokens={c_tok}, total_tokens={t_tok}"
                                        )
                                    fp = getattr(completion, 'system_fingerprint', None)
                                    logger.info(
                                        f"[TEXT2TEXT] ✅ (reduced tokens) {time.time()-start_time:.2f}s, len={len(ans)}"
                                        + (f", fingerprint={fp}" if fp else "")
                                    )
                                    self._last_text2text_error = None
                                    return ans
                                except Exception as e2:
                                    logger.error(f"[TEXT2TEXT] Reduced-token retry failed: {e2}")
                    logger.error(
                        f"[TEXT2TEXT] Retry {retry+1}: {e} | model={model_name} base_url={chosen_base}"
                    )
                    time.sleep(API_RETRY_DELAY)
            return ""
        except Exception as e:
            logger.error(f"[TEXT2TEXT] Failed: {e}")
            return ""

    # =========================================================================
    # VLM 批处理
    # =========================================================================

    def batch_video2text(self, task_li, batch_size=5, fps: Optional[float] = None):
        """
        【优化版】批量处理视频
        
        修复点：
        1. 默认 batch_size=5（一次处理所有视频）
        2. 修复内存判断逻辑（用总内存而非剩余内存）
        3. 添加帧数限制
        """
        if not task_li:
            return []
            
        start_time = time.time()
        
        # 【修复】根据 GPU 总内存决定 batch_size，而不是剩余内存
        if torch.cuda.is_available():
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            frame_counts = [len(img_paths) if img_paths is not None else 0 for _, img_paths, _ in task_li]
            avg_frames_per_video = sum(frame_counts) / max(len(frame_counts), 1)
            avg_frames_per_video = max(avg_frames_per_video, 1.0)
            if total_mem_gb >= 40:      # A100 40G/80G
                batch_size = min(5, max(1, int(200 / avg_frames_per_video)))
            elif total_mem_gb >= 24:    # 3090/4090
                batch_size = min(4, max(1, int(120 / avg_frames_per_video)))
            elif total_mem_gb >= 16:    # V100 16G
                batch_size = min(2, max(1, int(80 / avg_frames_per_video)))
            else:
                batch_size = 1
            logger.info(
                f"[BATCH] GPU: {total_mem_gb:.0f}GB, avg_frames: {avg_frames_per_video:.1f}, "
                f"batch_size: {batch_size}"
            )
        
        all_results = []
        
        for batch_start in range(0, len(task_li), batch_size):
            batch_end = min(batch_start + batch_size, len(task_li))
            batch_tasks = task_li[batch_start:batch_end]
            
            # 提取参数
            prompts = [t[0] for t in batch_tasks]
            image_paths_li = [t[1] for t in batch_tasks]
            timestamps_li = [t[2] for t in batch_tasks]
            
            batch_results = self._process_batch(prompts, image_paths_li, timestamps_li, fps=fps)
            all_results.extend(batch_results)
        
        logger.info(f'[BATCH] ✅ {len(task_li)} videos in {time.time()-start_time:.2f}s')
        return all_results
    
    def _process_batch(self, prompts, image_paths_li, timestamps_li, fps: Optional[float] = None):
        """处理单个批次"""
        if self.vlm_use_remote:
            return self._process_batch_remote(prompts, image_paths_li, fps=fps)
        batch_inputs = []
        debug_vlm = os.getenv("VLM_DEBUG", "0") == "1"

        use_image_frames_for_qwen = True if self.vlm_use_remote else (os.getenv("QWEN_USE_IMAGE_FRAMES", "1") == "1")

        for prompt, image_paths, timestamps in zip(prompts, image_paths_li, timestamps_li):
            image_data = []
            valid_paths = []

            for image_path in image_paths:
                if not os.path.exists(image_path):
                    continue
                try:
                    if self.model_type == 'internvl' or (self.model_type == 'qwen' and use_image_frames_for_qwen):
                        img = Image.open(image_path)
                        img.verify()
                        img = Image.open(image_path)
                        # 调整大小
                        w, h = img.size
                        if max(w, h) > MAX_IMAGE_DIM:
                            ratio = MAX_IMAGE_DIM / max(w, h)
                            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
                        image_data.append(img)
                    valid_paths.append(image_path)
                except Exception as e:
                    logger.warning(f"Invalid image: {image_path}, {e}")

            if (self.model_type == 'internvl' or (self.model_type == 'qwen' and use_image_frames_for_qwen)) and not image_data:
                batch_inputs.append(None)
                if debug_vlm:
                    logger.info("[VLM_DEBUG] image_data empty, skip input")
                continue
            if self.model_type != 'internvl' and not use_image_frames_for_qwen and not valid_paths:
                batch_inputs.append(None)
                if debug_vlm:
                    logger.info("[VLM_DEBUG] qwen valid_paths empty, skip input")
                continue

            # 构建消息
            if self.model_type == 'internvl' or (self.model_type == 'qwen' and use_image_frames_for_qwen):
                content = [{"type": "image", "image": img} for img in image_data]
                content.append({"type": "text", "text": "<image>\n" * len(image_data) + prompt})
            else:
                content = [{"type": "video", "video": valid_paths}, {"type": "text", "text": prompt}]

            messages = [{"role": "user", "content": content}]
            
            try:
                formatted_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = ("<image>\n" * len(image_data) + prompt) if self.model_type == 'internvl' else prompt

            fps = float(fps if fps is not None else getattr(self.args, "clip_fps", DEFAULT_CLIP_FPS))
            
            if self.model_type == 'internvl' or (self.model_type == 'qwen' and use_image_frames_for_qwen):
                multi_modal_data = {"image": image_data}
                mm_processor_kwargs = {}
            else:
                multi_modal_data = {"video": valid_paths}
                mm_processor_kwargs = {"min_pixels": 4*28*28, "max_pixels": 720*28*28, "fps": fps}

            if debug_vlm:
                path_count = len(image_data) if (self.model_type == "internvl" or (self.model_type == "qwen" and use_image_frames_for_qwen)) else len(valid_paths)
                mode = "image_frames" if (self.model_type == "qwen" and use_image_frames_for_qwen) else self.model_type
                logger.info(
                    f"[VLM_DEBUG] model={mode} paths={path_count} fps={fps} prompt_len={len(formatted_prompt)}"
                )

            batch_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": mm_processor_kwargs,
            })

        # 过滤无效输入
        valid_pairs = [(i, inp) for i, inp in enumerate(batch_inputs) if inp is not None]
        if not valid_pairs:
            return [""] * len(prompts)

        valid_indices, valid_inputs = zip(*valid_pairs)
        
        vlm_max_tokens = int(os.getenv('VLM_MAX_TOKENS', str(DEFAULT_VLM_MAX_TOKENS)))
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=vlm_max_tokens,
            seed=GLOBAL_SEED,
            repetition_penalty=1.2,
        )

        try:
            outputs = self.vlm_server.generate(list(valid_inputs), sampling_params=sampling_params, use_tqdm=False)
        except Exception as e:
            logger.error(f"vLLM generate failed: {e}")
            return [""] * len(prompts)

        results = [""] * len(prompts)
        for i, output in enumerate(outputs):
            results[valid_indices[i]] = self.remove_duplicate_sentences(output.outputs[0].text.strip())
            if debug_vlm:
                logger.info(
                    f"[VLM_DEBUG] output_len={len(results[valid_indices[i]])} idx={valid_indices[i]}"
                )

        return results

    def _process_batch_remote(self, prompts, image_paths_li, fps: Optional[float] = None):
        """通过远端 OpenAI 兼容接口调用 VLM（例如远端 vLLM serve）。"""
        debug_vlm = os.getenv("VLM_DEBUG", "0") == "1"
        results: List[str] = []

        # 限制传图数量，避免请求过大
        max_images = int(os.getenv("VLM_REMOTE_MAX_IMAGES", "32"))
        vlm_max_tokens = int(os.getenv('VLM_MAX_TOKENS', str(DEFAULT_VLM_MAX_TOKENS)))

        for prompt, image_paths in zip(prompts, image_paths_li):
            valid_paths = [p for p in (image_paths or []) if p and os.path.exists(p)]
            if not valid_paths:
                results.append("")
                if debug_vlm:
                    logger.info("[VLM_DEBUG][REMOTE] valid_paths empty, skip")
                continue

            valid_paths = valid_paths[:max_images]
            b64_frames = image_paths_to_base64(valid_paths)
            if not b64_frames:
                results.append("")
                if debug_vlm:
                    logger.info("[VLM_DEBUG][REMOTE] base64 encode failed/empty, skip")
                continue

            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                for b64 in b64_frames
            ]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            start_time = time.time()
            out_text = ""
            for retry in range(API_RETRY_COUNT):
                try:
                    completion = self.vlm_client.chat.completions.create(
                        model=self.vlm_model_name,
                        messages=messages,
                        temperature=0.0,
                        seed=GLOBAL_SEED,
                        max_completion_tokens=vlm_max_tokens,
                    )
                    out_text = (completion.choices[0].message.content or "").strip()
                    break
                except Exception as e:
                    logger.error(f"[VLM_REMOTE] Retry {retry+1}: {e} | model={self.vlm_model_name}")
                    time.sleep(API_RETRY_DELAY)

            out_text = self.remove_duplicate_sentences(out_text)
            results.append(out_text)

            if debug_vlm:
                logger.info(
                    f"[VLM_DEBUG][REMOTE] imgs={len(valid_paths)} out_len={len(out_text)} "
                    f"time={time.time()-start_time:.2f}s"
                )

        return results

    # =========================================================================
    # 答案解析
    # =========================================================================

    def _force_answer_from_evidence(
        self,
        dic: Dict,
        all_tool_results: Dict[str, str],
        answer_choices: List[str],
    ) -> Optional[str]:
        """当 LLM 持续返回空响应时，从已有工具证据中强制提取一个答案。

        策略：构造一条极精简的 prompt，只包含问题 + 选项 + 工具证据摘要，
        用更高的 max_tokens 单次调用 LLM；如果仍然失败则返回 None。
        """
        question = dic.get("question", "")
        options = dic.get("options", [])
        if not question or not answer_choices:
            return None

        # 构建证据摘要（每个视频最多600字符，保留结构化字段的关键信息）
        evidence_parts = []
        for vid, desc in all_tool_results.items():
            snippet = desc[:600] + ("..." if len(desc) > 600 else "")
            evidence_parts.append(f"Video {vid}: {snippet}")
        evidence_block = "\n".join(evidence_parts) if evidence_parts else "(no tool evidence)"

        prompt_version = str(getattr(self.args, "prompt_version", "default") or "default").lower()
        q_lower = question.lower()
        match_rule = ""
        if prompt_version == "short" and ("same action" in q_lower or "reference video" in q_lower):
            match_rule = (
                "Important: This is a reference-vs-candidates matching task. "
                "Prioritize ACTION consistency with the reference video over scene/background style. "
                "Do not pick by superficial similarity.\n"
            )

        force_prompt = (
            f"Question: {question}\n"
            f"Options:\n" + "\n".join(options) + "\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"{match_rule}"
            f"Based on the evidence above, which option is most likely correct? "
            f"Reply with ONLY <answer>X</answer> where X is one of {', '.join(answer_choices)}."
        )
        force_messages = [{"role": "user", "content": force_prompt}]

        # force_answer 只需输出 <answer>X</answer>，512 足够且不浪费配额
        old_env = os.environ.get("TEXT_MAX_COMPLETION_TOKENS")
        os.environ["TEXT_MAX_COMPLETION_TOKENS"] = "512"
        try:
            resp = self.single_text2text(
                force_messages, self.ds_model_name, self.ds_api_base, self.ds_api_keys
            )
        finally:
            if old_env is not None:
                os.environ["TEXT_MAX_COMPLETION_TOKENS"] = old_env
            else:
                os.environ.pop("TEXT_MAX_COMPLETION_TOKENS", None)

        if resp:
            answer = self.extract_final_answer(resp, allowed_choices=set(answer_choices))
            if answer and answer in set(answer_choices):
                logger.info(f"[FORCE_ANSWER] Extracted: {answer} from forced response")
                return answer
        logger.warning("[FORCE_ANSWER] Failed to extract answer from forced response")
        return None

    def extract_final_answer(self, text: str, allowed_choices: Optional[set] = None) -> str:
        """从模型输出中提取答案"""
        allowed = allowed_choices or ANSWER_CHOICES
        allowed_list = [str(x).strip() for x in (allowed or set()) if str(x).strip()]
        wants_letter = bool(allowed_list) and all(len(x) == 1 for x in allowed_list)
        allowed_lower_to_canon = {x.lower(): x for x in allowed_list}

        def _try_parse_letter(s: str):
            if not s or s.strip().lower() in {"[]", "null", "none", "nan", "", "n/a", "na"}:
                return None
            if re.match(r'^n\s*/?\s*a$', s, flags=re.I):
                return None
            
            s = RE_REMOVE_CODE_BLOCK.sub("", s).replace("```", "")
            
            for pat in (RE_PATTERN_ANSWER, RE_PATTERN_OPTION, RE_PATTERN_VIDEO):
                m = pat.search(s)
                if m:
                    for g in reversed(m.groups()):
                        if g and g.upper() in allowed:
                            return g.upper()

            # JSON 格式
            m = RE_JSON_ANSWER.search(s)
            if m:
                letter = m.groups()[-1].upper()
                return letter if letter in allowed else None

            # 最后一个独立字母
            s_cleaned = RE_NA.sub('', s.upper())
            cand = re.findall(r"\b([A-Z])\b", s_cleaned)
            cand = [c for c in cand if c in allowed]
            return cand[-1] if cand else None

        def _try_parse_word_choice(s: str) -> Optional[str]:
            if not s:
                return None
            s = RE_REMOVE_CODE_BLOCK.sub("", s).replace("```", "").strip()
            if not s:
                return None
            # If it contains <answer>...</answer>, prefer the last tag content.
            inner = RE_TAG_ANSWER.findall(RE_TAG_THINK.sub("", s))
            if inner:
                s = inner[-1].strip()
            low = s.strip().lower()
            if low in allowed_lower_to_canon:
                return allowed_lower_to_canon[low]
            # Accept common variants (e.g. WEAKENER / Strengthener.)
            low = re.sub(r"[\\s\\.;:!\\?\\)\\]\\}]+$", "", low)
            if low in allowed_lower_to_canon:
                return allowed_lower_to_canon[low]
            for k, v in allowed_lower_to_canon.items():
                if k and k in low:
                    return v
            return None

        try:
            text_cleaned = RE_TAG_THINK.sub('', text)
            answers = RE_TAG_ANSWER.findall(text_cleaned)
            
            for seg in reversed(answers):
                if wants_letter:
                    letter = _try_parse_letter(seg)
                    if letter:
                        return letter
                else:
                    w = _try_parse_word_choice(seg)
                    if w:
                        return w
            
            # 从尾部提取
            tail = text_cleaned[-200:] if len(text_cleaned) > 200 else text_cleaned
            if wants_letter:
                letter = _try_parse_letter(tail)
                return letter if letter else "-"
            w = _try_parse_word_choice(tail)
            return w if w else "-"
        except:
            return "-"

    # =========================================================================
    # 主处理流程
    # =========================================================================

    # === 新增功能入口：多粒度读取 + 对比式决策 + 不确定性触发 ===
    def _build_video_tasks(self, dic: Dict, question: str, video_ids: Optional[List[str]] = None, fps: Optional[float] = None):
        tasks = []
        for vid_id in video_ids or VIDEO_IDS:
            video_path = dic['video_map'].get(vid_id)
            duration = dic['duration_map'].get(vid_id, 10)
            if not video_path:
                continue
            video_clip, timestamps = timestamp_to_clip_path(
                self.full_dataset_folder, 0, duration, video_path, fps=(fps if fps is not None else self.args.clip_fps)
            )
            if video_clip:
                tasks.append((f"[Video {vid_id}] {question}", video_clip, timestamps, vid_id, ""))
        return tasks

    def _run_video_tasks(self, tasks, fps: Optional[float] = None):
        if not tasks:
            return []
        results = self.batch_video2text([(t[0], t[1], t[2]) for t in tasks], fps=fps)
        return [(tasks[i][3], results[i], tasks[i][4] if len(tasks[i]) > 4 else "") for i in range(len(results))]

    # === 新增功能：按时间段构建精读任务（粗扫 -> 细读）===
    def _build_video_tasks_for_ranges(
        self,
        dic: Dict,
        question: str,
        vid_ranges: Dict[str, List[tuple]],
        fps: Optional[float] = None,
    ):
        tasks = []
        for vid_id, ranges in (vid_ranges or {}).items():
            video_path = dic['video_map'].get(vid_id)
            duration = dic['duration_map'].get(vid_id, 10)
            if not video_path or not ranges:
                continue
            for begin, end in ranges:
                begin = max(0.0, float(begin))
                end = min(float(end), float(duration))
                if end - begin < 1.0:
                    continue
                video_clip, timestamps = timestamp_to_clip_path(
                    self.full_dataset_folder, begin, end, video_path,
                    fps=(fps if fps is not None else self.args.clip_fps)
                )
                if video_clip:
                    range_str = f"{begin:.1f}:{end:.1f}"
                    tasks.append((f"[Video {vid_id} {range_str}] {question}", video_clip, timestamps, vid_id, range_str))
        return tasks

    # === 新增功能：多粒度读取的帧率控制 ===
    def _get_coarse_fps(self) -> float:
        """粗读应该用更低的 FPS（0.3-1.0）快速扫描全局"""
        coarse = float(getattr(self.args, "coarse_fps", 0.5))
        return max(0.3, min(coarse, 1.0))

    def _get_fine_fps(self) -> float:
        """细读应该用更高的 FPS（1.0-4.0）捕捉细节"""
        fine = float(getattr(self.args, "fine_fps", 0.0))
        if fine <= 0:
            fine = float(getattr(self.args, "clip_fps", DEFAULT_CLIP_FPS))
        return max(1.0, min(fine, 4.0))

    # === 新增功能：对比式决策提示 ===
    # === 新增功能：不确定性检测 ===
    def _is_uncertain(self, text: str) -> bool:
        if not text:
            return False
        cleaned = RE_TAG_THINK.sub("", text)
        return bool(RE_UNCERTAIN.search(cleaned))

    def _extract_uncertain_videos(self, text: str, video_ids: Optional[List[str]] = None) -> List[str]:
        if not text:
            return []
        cleaned = RE_TAG_THINK.sub("", text)
        if not RE_UNCERTAIN.search(cleaned):
            return []
        lower = cleaned.lower()
        hits = set()
        ids = video_ids or list(VIDEO_IDS)
        for vid in ids:
            if re.search(rf"(^|[^A-Za-z]){re.escape(vid)}([^A-Za-z]|$)", cleaned):
                hits.add(vid)
            elif re.search(rf"\bvideo\s*{re.escape(vid.lower())}\b", lower):
                hits.add(vid)
            elif re.search(rf"\bcandidate\s*{re.escape(vid.lower())}\b", lower):
                hits.add(vid)
        return [vid for vid in ids if vid in hits]

    @staticmethod
    def _should_force_detection_for_comparison(question: str, options: List[str]) -> bool:
        text = (question or "") + " " + " ".join(options or [])
        t = text.lower()
        # 仅对“可度量属性”触发（颜色/数量）
        keywords = [
            "color", "colour", "red", "green", "yellow", "blue", "purple", "pink",
            "orange", "black", "white", "gray", "grey",
            "count", "number of", "how many", "total", "quantity",
        ]
        return any(k in t for k in keywords)

    @staticmethod
    def _should_force_detection_from_query(query: str) -> bool:
        """Use the actual video_reader question to decide if detection is needed."""
        if not query:
            return False
        t = query.lower()
        keywords = [
            # measurable attributes
            "color", "colour", "red", "green", "yellow", "blue", "purple", "pink",
            "orange", "black", "white", "gray", "grey",
            "count", "number of", "how many", "total", "quantity",
            # spatial/position cues
            "left", "right", "above", "below", "foreground", "background",
            "front", "back", "behind", "next to", "near", "position", "size",
        ]
        return any(k in t for k in keywords)

    def _find_uncertain_videos_in_tool_result(
        self,
        tool_result: str,
        video_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not tool_result:
            return []
        ids = video_ids or list(VIDEO_IDS)
        hits = []
        for vid in ids:
            section = self._extract_vid_section_from_block(tool_result, vid)
            if section and RE_UNCERTAIN.search(section):
                hits.append(vid)
        return hits

    def _collect_tool_evidence_for_video(self, tool_results: Dict[str, List[str]], vid_id: str) -> str:
        """收集 scene_graph / object_tracker 等确定性工具对某个视频的输出摘要。
        从整段工具输出中精确提取目标视频的部分，不截断。"""
        if not tool_results:
            return ""
        snippets: List[str] = []
        for tool_name in ("object_tracker", "scene_graph", "spatial_relation"):
            for block in tool_results.get(tool_name, []):
                # block 可能包含多个视频的结果，需要精确提取目标视频的部分
                extracted = self._extract_vid_section_from_block(block, vid_id)
                if extracted:
                    snippets.append(f"[{tool_name}] {extracted}")
        return "\n".join(snippets)

    @staticmethod
    def _extract_vid_section_from_block(block: str, vid_id: str) -> str:
        """从一段混合了多个视频的工具输出中，精确提取目标 vid_id 的部分。
        工具输出格式通常是：
          <tool id="X"> result:\n...\n<tool id="Y"> result:\n...
        或者只有 result 行（t=0.0s: ...）
        """
        if not block:
            return ""
        # 方法1：按 id="X" 标签分段
        # 找到 id="vid_id" 开始的位置，截取到下一个 id=" 或末尾
        marker = f'id="{vid_id}"'
        idx = block.find(marker)
        if idx == -1:
            return ""
        # 从 marker 开始，找 "result:" 或 ">"
        start = idx
        # 回退到行首
        line_start = block.rfind("\n", 0, start)
        start = line_start + 1 if line_start >= 0 else 0
        # 找到下一个 id=" 标签（不同视频）的位置作为结束
        rest = block[idx + len(marker):]
        next_id = rest.find('id="')
        if next_id >= 0:
            end = idx + len(marker) + next_id
        else:
            end = len(block)
        section = block[start:end].strip()
        # 提取 result: 之后的内容
        if "result:" in section:
            section = section.split("result:", 1)[1].strip()
        # 清理：去掉尾部可能的不完整标签
        section = re.sub(r'<[a-z_]+$', '', section).strip()
        if not section or section in ("disabled", "no detections", "no spatial relations detected", "no meaningful spatial relations"):
            return ""
        return section

    # === 新增功能：从规划文本解析精读时间段 ===
    def _parse_time_ranges(self, text: str, duration_map: Dict[str, float]) -> Dict[str, List[tuple]]:
        ranges = {vid: [] for vid in VIDEO_IDS}
        if not text:
            return ranges
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines:
            for vid in VIDEO_IDS:
                if not line.lower().startswith(vid.lower() + ":"):
                    continue
                content = line.split(":", 1)[1].strip()
                if not content or content.lower() in {"none", "n/a", "na"}:
                    continue
                pairs = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", content)
                for s, e in pairs[:2]:
                    begin, end = float(s), float(e)
                    dur = float(duration_map.get(vid, end))
                    begin = max(0.0, min(begin, dur))
                    end = max(0.0, min(end, dur))
                    if end - begin >= 1.0:
                        ranges[vid].append((begin, end))
        return ranges

    # =========================================================================
    # Direct VLM（Baseline 式直出）
    # =========================================================================

    # 每视频帧数预算：视频越多帧越少，总帧数控制在 ~30 以内以防爆显存
    _DIRECT_FRAME_BUDGET = {1: 20, 2: 14, 3: 10, 4: 8, 5: 6, 6: 4}
    _DIRECT_MAX_DIM = 448          # 与 eval_MLLM 对齐：448x448

    def _direct_vlm_answer(
        self,
        dic: Dict,
        answer_choices: List[str],
    ) -> Optional[str]:
        """Baseline 方式：所有视频帧 + question → VLM → 一步直出答案。

        复用已加载的 self.vlm_server（vLLM），以 image 模式发送帧，
        不占用 video slot（limit_mm={"video":1} 不受影响）。
        """
        video_ids = self._get_video_ids(dic)
        n_videos = len(video_ids)
        max_frames = self._DIRECT_FRAME_BUDGET.get(
            n_videos, 14 if n_videos <= 2 else max(2, 30 // n_videos)
        )
        max_dim = self._DIRECT_MAX_DIM

        def _to_square_448(img: Image.Image, size: int) -> Image.Image:
            """Align preprocessing with eval_MLLM: short-side resize + center crop square."""
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            short_side = min(w, h)
            if short_side <= 0:
                return img.resize((size, size), Image.Resampling.BICUBIC)
            scale = float(size) / float(short_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            left = max(0, (new_w - size) // 2)
            top = max(0, (new_h - size) // 2)
            right = min(left + size, new_w)
            bottom = min(top + size, new_h)
            img = img.crop((left, top, right, bottom))
            if img.size != (size, size):
                pad = Image.new("RGB", (size, size), color=(0, 0, 0))
                px = (size - img.size[0]) // 2
                py = (size - img.size[1]) // 2
                pad.paste(img, (px, py))
                img = pad
            return img

        all_images: List[Image.Image] = []
        vid_frame_map: List[tuple] = []          # (vid_id, n_frames)
        vid_images_map: Dict[str, List[Image.Image]] = {}

        for vid in video_ids:
            video_path = dic.get("video_map", {}).get(vid)
            if not video_path:
                continue
            duration = float(dic.get("duration_map", {}).get(vid, 10.0))
            end = max(1.0, duration)

            # 复用已有的帧提取函数
            try:
                frame_paths, _ts = timestamp_to_clip_path(
                    self.full_dataset_folder, 0.0, end, video_path,
                    fps=max(0.5, max_frames / end),
                )
            except Exception:
                continue
            if not frame_paths:
                continue

            # 均匀采样
            if len(frame_paths) > max_frames:
                idxs = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
                frame_paths = [frame_paths[i] for i in idxs]

            vid_images: List[Image.Image] = []
            for fp in frame_paths:
                try:
                    img = Image.open(fp)
                    img = _to_square_448(img, max_dim)
                    vid_images.append(img)
                except Exception:
                    continue

            if vid_images:
                vid_frame_map.append((vid, len(vid_images)))
                vid_images_map[vid] = vid_images
                all_images.extend(vid_images)

        if not all_images:
            logger.warning("[DIRECT_VLM] No frames extracted, skip")
            return None

        # ── 构造 prompt ──
        question = dic.get("question", "")
        options = dic.get("options", [])
        opt_text = "\n".join(options) if isinstance(options, list) else str(options)
        video_text_map = dic.get("video_text_map", {}) or {}

        # 帧 → 视频映射标签
        label_lines: List[str] = []
        cursor = 1
        for vid, nf in vid_frame_map:
            label_lines.append(f"Images {cursor}-{cursor + nf - 1} are from Video {vid}")
            cursor += nf

        # CR 等任务：把每个视频自带的 question/option 也明确传给 direct VLM
        per_video_text_lines: List[str] = []
        if isinstance(video_text_map, dict) and video_text_map:
            ordered_ids = [vid for vid, _ in vid_frame_map]
            for vid in ordered_ids:
                text_info = video_text_map.get(vid, {})
                if not isinstance(text_info, dict):
                    continue
                q = str(text_info.get("cf_question", "")).strip()
                o1 = str(text_info.get("cf_option1", "")).strip()
                o2 = str(text_info.get("cf_option2", "")).strip()
                if not (q or o1 or o2):
                    continue
                per_video_text_lines.append(f"[Video {vid}]")
                if q:
                    per_video_text_lines.append(f"Question: {q}")
                if o1:
                    per_video_text_lines.append(f"Option 1: {o1}")
                if o2:
                    per_video_text_lines.append(f"Option 2: {o2}")
            if per_video_text_lines:
                per_video_text_lines = ["Per-video textual context:"] + per_video_text_lines

        per_video_text_block = "\n".join(per_video_text_lines) + "\n\n" if per_video_text_lines else ""
        prompt_text = (
            "\n".join(label_lines) + "\n\n"
            f"Question: {question}\n{opt_text}\n\n"
            + per_video_text_block
            + f"Look at ALL the video frames carefully. "
            + f"Select the correct answer from the options. "
            + f"Reply with ONLY <answer>X</answer> where X is one of {', '.join(answer_choices)}."
        )

        if self.vlm_use_remote:
            # ── 远端 VLM：走 OpenAI-compatible API（base64 image） ──
            import base64, io
            per_video_frames = int(os.getenv("DIRECT_REMOTE_FRAMES_PER_VIDEO", "4"))
            max_images_remote_raw = int(os.getenv("DIRECT_REMOTE_MAX_IMAGES", "0"))
            # <=0 表示不设总图上限；仅在显式设置 >0 时启用截断
            max_images_remote = max_images_remote_raw if max_images_remote_raw > 0 else None
            min_images_remote = int(os.getenv("DIRECT_REMOTE_MIN_IMAGES", "2"))
            min_images_remote = max(1, min_images_remote)
            per_video_frames = max(1, per_video_frames)

            # Align with eval_MLLM style: cap frames per video first.
            selected_images: List[Image.Image] = []
            for vid in video_ids:
                imgs = vid_images_map.get(vid, [])
                if not imgs:
                    continue
                if len(imgs) > per_video_frames:
                    idxs = np.linspace(0, len(imgs) - 1, per_video_frames, dtype=int)
                    imgs = [imgs[i] for i in idxs]
                selected_images.extend(imgs)
            if not selected_images:
                selected_images = list(all_images)
            if max_images_remote is not None and len(selected_images) > max_images_remote:
                idxs = np.linspace(0, len(selected_images) - 1, max_images_remote, dtype=int)
                selected_images = [selected_images[i] for i in idxs]

            response = ""
            while True:
                api_content: List[Dict[str, Any]] = []
                for img in selected_images:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=75, optimize=True)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    api_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                api_content.append({"type": "text", "text": prompt_text})
                api_msg = [{"role": "user", "content": api_content}]

                try:
                    completion = self.vlm_client.chat.completions.create(
                        model=self.vlm_model_name,
                        messages=api_msg,
                        temperature=0.0,
                        max_tokens=128,
                        seed=GLOBAL_SEED,
                    )
                    response = completion.choices[0].message.content.strip()
                    break
                except Exception as e:
                    err = str(e)
                    too_long = ("maximum model length" in err) or ("decoder prompt" in err) or ("max_model_len" in err)
                    if too_long and len(selected_images) > min_images_remote:
                        new_n = max(min_images_remote, len(selected_images) // 2)
                        if new_n == len(selected_images):
                            new_n = len(selected_images) - 1
                        idxs = np.linspace(0, len(selected_images) - 1, new_n, dtype=int)
                        selected_images = [selected_images[i] for i in idxs]
                        logger.warning(
                            f"[DIRECT_VLM] Remote prompt too long, retry with fewer images: {len(idxs)}"
                        )
                        continue
                    logger.error(f"[DIRECT_VLM] Remote generate failed: {e}")
                    return None
        else:
            # ── 本地 vLLM：image 模式 ──
            content = [{"type": "image", "image": img} for img in all_images]
            content.append({"type": "text", "text": "<image>\n" * len(all_images) + prompt_text})
            msg = [{"role": "user", "content": content}]

            try:
                formatted_prompt = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                formatted_prompt = "<image>\n" * len(all_images) + prompt_text

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=128,
                seed=GLOBAL_SEED,
                repetition_penalty=1.2,
            )

            try:
                outputs = self.vlm_server.generate(
                    [{
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"image": all_images},
                        "mm_processor_kwargs": {},
                    }],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                response = outputs[0].outputs[0].text.strip()
            except Exception as e:
                logger.error(f"[DIRECT_VLM] Local generate failed: {e}")
                return None

        logger.info(f"[DIRECT_VLM] n_videos={n_videos}, n_frames={len(all_images)}, "
                     f"response_len={len(response)}")

        answer = self.extract_final_answer(response, allowed_choices=set(answer_choices))
        if answer and answer in set(answer_choices):
            logger.info(f"[DIRECT_VLM] Extracted answer: {answer}")
            return answer

        # 如果 extract_final_answer 解析失败，尝试直接取首字母
        raw = response.strip().upper()
        if len(raw) >= 1 and raw[0] in set(answer_choices):
            logger.info(f"[DIRECT_VLM] Raw first-char answer: {raw[0]}")
            return raw[0]

        logger.warning(f"[DIRECT_VLM] Could not parse answer from: {response[:200]}")
        return None

    def _normalize_answer_text(self, s: Any, answer_choices: Optional[List[str]] = None) -> str:
        """Normalize an answer string for comparison/scoring."""
        if s is None:
            return ""
        if isinstance(s, list):
            s = s[0] if s else ""
        s = str(s).strip()
        if not s:
            return ""
        if answer_choices:
            try:
                parsed = self.extract_final_answer(s, allowed_choices=set(answer_choices))
                if parsed and parsed != "-":
                    return str(parsed).strip()
            except Exception:
                pass
        return s

    def _answers_equal(self, a: str, b: str, answer_choices: Optional[List[str]] = None) -> bool:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return False
        is_letter = bool(answer_choices) and all(
            isinstance(c, str) and len(c.strip()) == 1 for c in (answer_choices or [])
        )
        if is_letter:
            return a.upper() == b.upper()
        return a.casefold() == b.casefold()

    def _score_prediction(self, dic: Dict, pred: Any, answer_choices: List[str]) -> tuple:
        """Return (pred_norm, gt_norm, score_bool)."""
        gt_norm = self._normalize_answer_text(dic.get("answer", ""), answer_choices)
        pred_norm = self._normalize_answer_text(pred, answer_choices)
        return pred_norm, gt_norm, self._answers_equal(pred_norm, gt_norm, answer_choices)

    def _finalize_result(self, dic, answer, all_descriptions, question_history, num_rounds, messages, tool_results=None):
        """打包最终结果"""
        answer_choices = self._get_answer_choices(dic)
        pred_norm, gt_norm, score = self._score_prediction(dic, answer, answer_choices)
        dic['pred_answer'] = pred_norm
        self.experience_pool.append({
            'question': dic.get('question', ''),
            'gt_answer': gt_norm,
            'pred_answer': pred_norm,
            'score': score,
            'analyzed_videos': list(all_descriptions.keys()),
            'video_descriptions': dict(all_descriptions),
            'num_rounds': num_rounds,
            'questions_asked': question_history,
            'tool_results': tool_results or {},
        })
        
        return {
            'messages': messages,
            'raw_data': dic,
            'score': score,
            'video_descriptions': dict(all_descriptions),
        }


    def process_single_sample(self, dic: Dict) -> Dict:
        """标准流程处理"""
        video_ids = self._get_video_ids(dic)
        answer_choices = self._get_answer_choices(dic)
        answer_set = set(answer_choices)
        eval_stats = dic.get("eval_stats")
        if not isinstance(eval_stats, dict):
            eval_stats = {}
        eval_stats.setdefault("tool_call_count", 0)
        eval_stats.setdefault("tool_call_turns", 0)
        eval_stats.setdefault("invalid_turns", 0)
        dic["eval_stats"] = eval_stats
        torch.cuda.empty_cache()
        _task_for_asr = dic.get("task", "") or getattr(self.args, "dataset", "")
        if self.args.use_subtitle or (
            self.asr_enable_defeasible
            and is_defeasible_entailment_task(_task_for_asr)
        ):
            dic = self.get_dic_subtitles(dic)

        n_videos = len(video_ids)
        direct_threshold = int(getattr(self.args, "direct_vlm_threshold", 0))

        # ================================================================
        # Gate A: Direct-first — 视频数 ≤ threshold 时先跑 baseline 直出
        #   成功 → 立即返回（省掉整个 agent 循环）
        #   失败 → 落入下面的 agent 流程
        # ================================================================
        _direct_answer_cache: Optional[str] = None      # 缓存结果供后续 fallback/hint
        if direct_threshold > 0:
            logger.info(
                f"[DIRECT_VLM_GATE] n_videos={n_videos}, threshold={direct_threshold}"
            )
            _direct_answer_cache = self._direct_vlm_answer(dic, answer_choices)
            if _direct_answer_cache and _direct_answer_cache in answer_set:
                if n_videos <= direct_threshold:
                    # 视频少，direct 可信度高 → 直接返回
                    pred_norm, gt_norm, score = self._score_prediction(dic, _direct_answer_cache, answer_choices)
                    dic['pred_answer'] = pred_norm
                    logger.info(
                        f"[DIRECT_VLM_GATE] Direct-first accepted: {pred_norm} "
                        f"(gt={gt_norm}, correct={score})"
                    )
                    if 'subtitles' in dic:
                        del dic['subtitles']
                    return {
                        'messages': [{'role': 'system', 'content': f'[direct_vlm] answer={_direct_answer_cache}'}],
                        'raw_data': dic,
                        'score': score,
                        'video_descriptions': {},
                    }
                # 视频多（> threshold）：direct 作为 hint 注入 agent prompt
                # （不直接返回，让 agent 验证）

        original_task = dic.get('question', '')
        messages = []
        initial_prompt = self.build_initial_prompt(dic)

        # Skills → system message（attention权重更高，不被tool results冲淡）
        if self.skills_text:
            self._append_message(messages, "system", self.skills_text)

        # Gate B: 把 direct answer 作为 Visual Hint 注入 agent prompt
        if _direct_answer_cache and _direct_answer_cache in answer_set and n_videos > direct_threshold:
            initial_prompt += (
                f"\n\n## Visual Hint (from direct frame inspection)\n"
                f"A quick visual scan of all video frames suggests the answer "
                f"might be **{_direct_answer_cache}**.\n"
                f"Use this as a reference point. Verify with video_reader evidence "
                f"and override if your detailed analysis disagrees.\n"
            )

        self._append_message(messages, "user", [{"type": "text", "text": initial_prompt}])
        
        cur_turn = 0
        analyzed_videos = set()
        failed_videos = set()
        empty_read_fail_counts: Dict[str, int] = {}
        all_tool_results: Dict[str, str] = {}
        tool_results: Dict[str, List[str]] = {}
        if dic.get("no_ref"):
            focus = set(self._focus_video_ids_by_question(dic, video_ids))
            if focus and len(focus) < len(video_ids):
                required_videos = focus
            else:
                required_videos = set(video_ids)
        else:
            required_videos = set(video_ids)

        # Optional: run tools once upfront to give the planner high-level evidence.
        self._run_pretool_sweep(dic, messages=messages, tool_cache=all_tool_results, tool_results=tool_results)

        _empty_retries = 0          # 空响应连续重试计数器
        _MAX_EMPTY_RETRIES = 2       # 最多连续重试2次（压缩context后重试）
        _postproc_empty_retries = 0  # 清洗/约束后变空的连续重试计数器（避免死循环）
        _MAX_POSTPROC_EMPTY_RETRIES = int(os.getenv("POSTPROC_EMPTY_MAX_RETRIES", "3"))
        _short_max_empty_read_retries = int(os.getenv("SHORT_MAX_EMPTY_READ_RETRIES", "3"))
        _max_empty_read_retries_general = int(os.getenv("EMPTY_READ_MAX_RETRIES", "2"))

        while cur_turn < MAX_DS_ROUND:
            self._maybe_compress_context_short(
                messages=messages,
                dic=dic,
                all_tool_results=all_tool_results,
                tool_results=tool_results,
                video_ids=video_ids,
                cur_turn=cur_turn,
            )
            raw_output_text = self.single_text2text(messages, self.ds_model_name, self.ds_api_base, self.ds_api_keys)
            output_text = raw_output_text
            prompt_version = str(getattr(self.args, "prompt_version", "default") or "default").lower()
            if prompt_version == "short":
                output_text = self._strip_think_content(output_text)
            else:
                pass
            if not output_text:
                # 仅 short：模型有返回，但清洗后变空，属于格式污染，不应立刻走空答兜底
                if (
                    prompt_version == "short"
                    and isinstance(raw_output_text, str)
                    and raw_output_text.strip()
                ):
                    eval_stats["invalid_turns"] += 1
                    logger.warning("[EMPTY_AFTER_CLEAN] non-empty raw response became empty after cleaning; reprompt planner")
                    self._append_message(
                        messages,
                        "user",
                        "Your previous response became empty after format cleanup. "
                        "Output EXACTLY one tool call + [Pause] OR final <answer>X</answer>.",
                    )
                    continue
                if self._last_text2text_error == "max_tokens_too_large":
                    logger.warning("[DIRECT_VLM_FAST_FALLBACK] text2text hit max_tokens limit, fallback to direct VLM")
                    fast_fallback = self._direct_vlm_answer(dic, answer_choices)
                    if fast_fallback and fast_fallback in answer_set:
                        pred_norm, gt_norm, score = self._score_prediction(dic, fast_fallback, answer_choices)
                        dic['pred_answer'] = pred_norm
                        if 'subtitles' in dic:
                            del dic['subtitles']
                        return {
                            'messages': messages,
                            'raw_data': dic,
                            'score': score,
                            'video_descriptions': dict(all_tool_results),
                        }
                eval_stats["invalid_turns"] += 1
                _empty_retries += 1
                logger.warning(
                    f"[EMPTY_RESPONSE] Turn {cur_turn}, empty_retry {_empty_retries}/{_MAX_EMPTY_RETRIES}"
                )
                if _empty_retries <= _MAX_EMPTY_RETRIES:
                    # ── 策略：压缩 context 后重试 ──
                    # 保留第一条 user message（initial prompt）和最后2条 message，
                    # 中间的 pretool / tool evidence 被截断压缩。
                    if len(messages) > 4:
                        kept_head = messages[:1]          # initial prompt
                        kept_tail = messages[-2:]         # 最近的交互
                        mid_summary = (
                            "[System] Previous tool evidence has been summarized to save context. "
                            "Please analyze the question and available evidence, then either "
                            "call a tool or provide <answer>X</answer>."
                        )
                        messages[:] = kept_head + [{"role": "user", "content": mid_summary}] + kept_tail
                        logger.info(
                            f"[EMPTY_RESPONSE] Compressed context: {len(messages)} messages remain"
                        )
                    continue
                else:
                    # ── 所有重试用尽 → force_answer：从已有证据中强制提取答案 ──
                    logger.warning("[EMPTY_RESPONSE] All retries exhausted, forcing answer from evidence")
                    force_answer = self._force_answer_from_evidence(
                        dic, all_tool_results, answer_choices
                    )
                    if force_answer and force_answer in answer_set:
                        pred_norm, gt_norm, score = self._score_prediction(dic, force_answer, answer_choices)
                        dic['pred_answer'] = pred_norm
                        if 'subtitles' in dic:
                            del dic['subtitles']
                        return {
                            'messages': messages,
                            'raw_data': dic,
                            'score': score,
                            'video_descriptions': dict(all_tool_results),
                        }
                    break   # force_answer 也失败了，退出
            else:
                _empty_retries = 0    # 成功响应，重置计数器

            pending_videos = required_videos - analyzed_videos - failed_videos
            is_short = str(getattr(self.args, "prompt_version", "default")).lower() == "short"
            short_require_all = os.getenv("SHORT_REQUIRE_ALL_READ", "1") == "1"
            sim_ready, sim_margin, sim_top_vid, sim_scores = self._similarity_rank_signal(
                dic, required_videos - failed_videos, tool_results
            )
            if self._is_similarity_centric_question(dic) and sim_ready:
                # Similarity tasks: if visual_similarity ranking is complete and confident,
                # do not force extra video_reader rounds.
                pending_videos = set()
            if is_short and short_require_all:
                # short 模式默认严格：未读完所有候选前，不接受最终答案
                can_answer = (len(pending_videos) == 0)
            else:
                can_answer = (cur_turn >= MAX_DS_ROUND - 3) or (len(pending_videos) == 0)

            # Planner 不确定时不再硬编码注入 video_reader。
            # 保留 planner 原始输出，由后续 user message 引导选择工具。
            
            if '<answer>' in output_text and cur_turn < MAX_DS_ROUND - 1:
                recent_reader_ids = RE_VIDEO_READER_ID.findall(output_text)
                recent_evidence = "\n".join(
                    all_tool_results.get(v, "") for v in recent_reader_ids
                )
                options = dic.get("options", []) if dic else []
                if options and recent_evidence:
                    option_hits = 0
                    for opt in options:
                        distinguishing = ""
                        if "with " in opt:
                            distinguishing = opt.split("with ", 1)[-1]
                        elif ". " in opt:
                            distinguishing = opt.split(". ", 1)[-1]
                        keywords = [w for w in distinguishing.lower().split() if len(w) > 3]
                        if any(kw in recent_evidence.lower() for kw in keywords):
                            option_hits += 1
                    if option_hits >= 2:
                        output_text = re.sub(r'<answer>.*?</answer>', '', output_text, flags=re.DOTALL).strip()
                        if not output_text.strip():
                            vid = (recent_reader_ids[0] if recent_reader_ids else (video_ids[0] if video_ids else 'ref'))
                            dur = dic.get('duration_map', {}).get(vid, 10) if dic else 10
                            a0 = options[0].split(". ", 1)[-1][:60] if len(options) > 0 else "option A"
                            a1 = options[1].split(". ", 1)[-1][:60] if len(options) > 1 else "option B"
                            output_text = (
                                f'<video_reader id="{vid}">0:{dur}</video_reader>'
                                f'<video_reader_question>'
                                f'The previous description was ambiguous. '
                                f'Focus on the SINGLE most critical difference: {a0} vs {a1}. '
                                f'Describe exactly what you see.'
                                f'</video_reader_question>'
                            )
            # 如果 planner 在同一轮既调了工具又给了答案，删掉答案（先收齐证据再推理）
            has_tool_call = any(tag in output_text for tag in (
                '<video_reader', '<scene_graph', '<object_tracker',
                '<scene_detector', '<spatial_relation', '<visual_similarity',
                '<temporal_grounding_agent', '<subtitle_retriever', '<subtitle_extractor',
            ))
            if has_tool_call and '<answer>' in output_text:
                output_text = re.sub(r'<answer>.*?</answer>', '', output_text, flags=re.DOTALL).strip()

            if not can_answer and '<answer>' in output_text:
                output_text = re.sub(r'<answer>.*?</answer>', '', output_text, flags=re.DOTALL).strip()
                # 不再硬编码 video_reader 调用，让后续 user message 引导 planner 选择工具

            if not output_text.strip():
                eval_stats["invalid_turns"] += 1
                _postproc_empty_retries += 1
                logger.warning(
                    "[INVALID_TURN] Empty assistant output before tool parsing; "
                    f"postproc_empty_retry={_postproc_empty_retries}/{_MAX_POSTPROC_EMPTY_RETRIES}"
                )
                if _postproc_empty_retries <= _MAX_POSTPROC_EMPTY_RETRIES:
                    # 更强约束：要求输出工具调用而不是答案，避免反复输出 <answer> 被剥离后变空
                    self._append_message(
                        messages,
                        "user",
                        "Your previous response became empty after enforcing constraints. "
                        "You are NOT allowed to answer yet. Output EXACTLY ONE valid tool call "
                        "(e.g., <video_reader ...></video_reader><video_reader_question>...</video_reader_question>) "
                        "and nothing else. Do NOT output <answer>.",
                    )
                    continue

                # 多次 reprompt 仍为空 → 自动注入一个最保守的 video_reader 工具调用，推进流程，避免死循环
                candidate_vids = list(pending_videos) if isinstance(pending_videos, set) else []
                vid = (
                    "ref"
                    if "ref" in pending_videos
                    else (candidate_vids[0] if candidate_vids else (video_ids[0] if video_ids else "ref"))
                )
                dur = (dic.get("duration_map", {}) or {}).get(vid, 10) if dic else 10
                # video_reader 支持小数；这里取一个短窗口作为最保守的推进
                end_t = int(max(1, min(20, float(dur) if dur else 10)))
                output_text = (
                    f'<video_reader id="{vid}">0:{end_t}</video_reader>'
                    f'<video_reader_question>'
                    f'Describe the key objects/actions in this segment. Focus on evidence needed to answer the task.'
                    f'</video_reader_question>'
                )
                logger.warning(f"[INVALID_TURN_FALLBACK] Injected safe video_reader call: vid={vid} 0:{end_t}")
            else:
                _postproc_empty_retries = 0  # 成功产出非空，重置该计数器

            # ── Prune overly dense tool calls (especially for counting) ──
            # Some planners emit dozens of segmented video_reader calls (0:12,12:24,...)
            # This bloats context and can cause repeated empty reads. We keep a compact,
            # executable subset before appending to messages and parsing tools.
            def _is_counting_like(_dic: Dict) -> bool:
                _task = _dic.get("task", "") if _dic else ""
                if isinstance(_task, list):
                    _task = _task[0] if _task else ""
                if _task == "Counting":
                    return True
                q = str((_dic.get("question", "") if _dic else "")).lower()
                return bool(re.search(r"\b(how\s+many|count|number\s+of|same\s+number|same\s+count|equal\s+number|as\s+many)\b", q))

            if _is_counting_like(dic):
                # Hard caps to control cost: keep tool calls small and windows coarse.
                # - First turn: at most 5 tool blocks total, and keep ONLY 1 video_reader block
                #   (prefer ref) to satisfy the "first-turn video_reader" constraint.
                # - Later turns: keep small per-video caps.
                max_vr_per_vid = int(os.getenv("COUNTING_VR_MAX_PER_VIDEO", "1"))
                max_sg_per_vid = int(os.getenv("COUNTING_SG_MAX_PER_VIDEO", "1"))
                max_total_blocks = int(os.getenv("COUNTING_MAX_TOOL_BLOCKS_PER_TURN", "20"))
                first_turn_max_blocks = int(os.getenv("COUNTING_FIRST_TURN_MAX_TOOL_BLOCKS", "5"))
                first_turn_max_vr_total = int(os.getenv("COUNTING_FIRST_TURN_MAX_VR_BLOCKS", "1"))

                vr_seen: Dict[str, int] = {}
                sg_seen: Dict[str, int] = {}
                kept_blocks: List[str] = []
                kept_vr_total = 0

                block_pat = re.compile(
                    r'(<video_reader id="([^"]+)">[\s\S]*?</video_reader>\s*<video_reader_question>[\s\S]*?</video_reader_question>)'
                    r'|(<scene_graph id="([^"]+)">[\s\S]*?</scene_graph>)',
                    flags=re.I,
                )

                # First pass: if first turn, try to keep ONE ref video_reader block first.
                if cur_turn == 0:
                    for m in block_pat.finditer(output_text):
                        vr_block, vr_vid = m.group(1), m.group(2)
                        if not (vr_block and vr_vid):
                            continue
                        v = self._resolve_video_id(vr_vid, dic)
                        if v == "ref":
                            kept_blocks.append(vr_block)
                            kept_vr_total = 1
                            vr_seen[v] = 1
                            break

                # Second pass: keep remaining blocks under caps.
                for m in block_pat.finditer(output_text):
                    vr_block, vr_vid = m.group(1), m.group(2)
                    sg_block, sg_vid = m.group(3), m.group(4)

                    # dynamic caps for first turn
                    cur_max_total = first_turn_max_blocks if cur_turn == 0 else max_total_blocks
                    cur_max_vr_total = first_turn_max_vr_total if cur_turn == 0 else 10**9

                    if len(kept_blocks) >= cur_max_total:
                        break

                    if vr_block and vr_vid:
                        v = self._resolve_video_id(vr_vid, dic)
                        if cur_turn == 0 and kept_vr_total >= cur_max_vr_total:
                            continue
                        # If we already kept the ref block in the first-pass, skip duplicates.
                        if cur_turn == 0 and v == "ref" and vr_seen.get(v, 0) >= 1:
                            continue
                        vr_seen[v] = vr_seen.get(v, 0) + 1
                        if vr_seen[v] <= max_vr_per_vid:
                            kept_blocks.append(vr_block)
                            kept_vr_total += 1
                        continue

                    if sg_block and sg_vid:
                        v = self._resolve_video_id(sg_vid, dic)
                        sg_seen[v] = sg_seen.get(v, 0) + 1
                        if sg_seen[v] <= max_sg_per_vid:
                            kept_blocks.append(sg_block)

                if kept_blocks:
                    output_text = "\n".join(kept_blocks).strip()

            turn_tool_calls = self._count_tool_calls_in_text(output_text)
            if turn_tool_calls > 0:
                eval_stats["tool_call_turns"] += 1
                eval_stats["tool_call_count"] += turn_tool_calls

            self._append_message(messages, "assistant", output_text)
            cur_turn += 1

            tool_result = self.process_tool_calls(output_text, dic, all_tool_results, tool_results)

            # repeated empty segment：达到阈值后放弃，避免无效重试循环。
            if tool_result:
                empty_vids = re.findall(r"Video\s+([A-Za-z0-9_]+)\s+at\s+[^:]+:\s+empty segment\.", tool_result)
                for vid in empty_vids:
                    resolved_vid = self._resolve_video_id(vid, dic)
                    empty_read_fail_counts[resolved_vid] = empty_read_fail_counts.get(resolved_vid, 0) + 1
                    prompt_now = str(getattr(self.args, "prompt_version", "default")).lower()
                    limit = _short_max_empty_read_retries if prompt_now == "short" else _max_empty_read_retries_general
                    if empty_read_fail_counts[resolved_vid] >= limit:
                        failed_videos.add(resolved_vid)
                        logger.warning(
                            f"[SKIP_EMPTY_VIDEO] video={resolved_vid} "
                            f"empty_segment_retries={empty_read_fail_counts[resolved_vid]} "
                            f">= {limit}, mark as failed and skip further reads"
                        )
            
            video_reader_ids = RE_VIDEO_READER_ID.findall(output_text)
            # 仅当该视频确实拿到可用证据时，才标记为已读。
            # 避免“调用过 video_reader 但返回 empty segment”也被算作已读。
            for vid in video_reader_ids:
                if (all_tool_results.get(vid) or "").strip():
                    analyzed_videos.add(vid)

            # ── 交叉验证：确定性工具证据 + 属性冲突检测 ──
            if tool_result and video_reader_ids and tool_results:
                cross_ref_notes: List[str] = []
                all_conflicts: Dict[str, List[Dict]] = {}   # vid → conflicts
                reread_calls: List[str] = []                 # 待 re-read 的 video_reader 调用

                for vid in video_reader_ids:
                    evidence = self._collect_tool_evidence_for_video(tool_results, vid)
                    if not evidence:
                        continue

                    # 1) 原有证据附注
                    cross_ref_notes.append(
                        f"⚠️ Video {vid} tool evidence:\n{evidence}"
                    )

                    # 2) 通用属性冲突检测（仅在 ENABLE_ATTRIBUTE_OVERRIDE 开启时生效）
                    if ENABLE_ATTRIBUTE_OVERRIDE:
                        vlm_text = all_tool_results.get(vid, "")
                        sg_ev = "\n".join(
                            self._extract_vid_section_from_block(b, vid)
                            for b in tool_results.get("scene_graph", [])
                        )
                        ot_ev = "\n".join(
                            self._extract_vid_section_from_block(b, vid)
                            for b in tool_results.get("object_tracker", [])
                        )
                        conflicts = _detect_attribute_conflicts(vlm_text, sg_ev, ot_ev)
                        if conflicts:
                            all_conflicts[vid] = conflicts
                            for c in conflicts:
                                cross_ref_notes.append(
                                    f"⚠️ CONFLICT Video {vid} [{c['type']}]: "
                                    f"VLM says {c['vlm_val']}, detection says {c['det_val']} "
                                    f"({c['note']}). "
                                    f"Re-read this video to resolve."
                                )

                # 3) 可选 re-read：两种模式
                if CONFLICT_REREAD and all_conflicts:
                    # 模式 A (CONFLICT_REREAD = 'auto')：
                    #   代码直接构造 video_reader 调用并执行，不经过 LLM。
                    #   快但问题是硬编码的。
                    # 模式 B (CONFLICT_REREAD = 'llm', 默认)：
                    #   只在 annotation 里引导 LLM，由 LLM 自己写 re-read 问题。
                    #   LLM 知道原始问题上下文，能写出更针对性的 question。
                    if CONFLICT_REREAD == 'auto':
                        for vid, clist in all_conflicts.items():
                            dur = dic.get('duration_map', {}).get(vid, 10)
                            hints = "; ".join(
                                f"{c['type']}={c['det_val']}" for c in clist
                            )
                            reread_calls.append(
                                f'<video_reader id="{vid}">0:{dur}</video_reader>'
                                f'<video_reader_question>'
                                f'Detection tools found: {hints}. '
                                f'Look very carefully and verify these attributes. '
                                f'Report exactly what you see.'
                                f'</video_reader_question>'
                            )
                        if reread_calls:
                            reread_text = "\n".join(reread_calls)
                            reread_result = self.process_video_reader(
                                reread_text, dic, all_tool_results
                            )
                            if reread_result.strip():
                                tool_result += f"\n🔄 Re-read (auto):\n{reread_result}"
                                self._append_tool_result(tool_results, "video_reader", reread_result)
                    else:
                        # 模式 B：引导 LLM 自己写 re-read
                        # 在 annotation 里明确告诉 LLM 该 re-read 哪些视频、验证什么
                        reread_hints: List[str] = []
                        for vid, clist in all_conflicts.items():
                            attrs = ", ".join(
                                f"{c['type']}(detection={c['det_val']})" for c in clist
                            )
                            reread_hints.append(
                                f"Re-read Video {vid} to verify: {attrs}. "
                                f"Ask a specific question about the conflicting attribute."
                            )
                        cross_ref_notes.append(
                            "🔄 " + " ".join(reread_hints)
                        )

                # 4) 不覆盖 VLM 证据，仅追加冲突提示
                if ENABLE_ATTRIBUTE_OVERRIDE:
                    for vid, clist in all_conflicts.items():
                        for c in clist:
                            existing = all_tool_results.get(vid, "")
                            if existing:
                                all_tool_results[vid] = (
                                    existing + f" [⚠ conflict: VLM={c['vlm_val']}, det={c['det_val']}]"
                                )

                # 5) 汇总：证据 + 冲突 + 信任指南
                if cross_ref_notes:
                    tool_result += "\n" + "\n".join(cross_ref_notes)
                    if all_conflicts:
                        tool_result += f"\n{TRUST_GUIDE}"

            if tool_result:
                pending_videos = required_videos - analyzed_videos - failed_videos
                sim_ready, sim_margin, sim_top_vid, sim_scores = self._similarity_rank_signal(
                    dic, required_videos - failed_videos, tool_results
                )
                if self._is_similarity_centric_question(dic) and sim_ready:
                    pending_videos = set()
                    tool_result += (
                        f"\n✅ Similarity ranking confident "
                        f"(margin={sim_margin:.3f}, top={sim_top_vid}). "
                        f"You may answer now without additional reads."
                    )
                if self._handle_tool_result(
                    tool_result,
                    pending_videos,
                    cur_turn,
                    all_tool_results,
                    messages,
                    dic=dic,
                    video_ids=video_ids,
                    tool_results=tool_results,
                ):
                    continue

            if '<answer>' in output_text:
                answer = self.extract_final_answer(output_text, allowed_choices=answer_set)
                if answer not in answer_set:
                    self._append_message(messages, "user", self._invalid_answer_prompt(answer_choices))
                    continue
                
                dic['pred_answer'] = answer
                pred_norm, gt_norm, score = self._score_prediction(dic, answer, answer_choices)
                dic['pred_answer'] = pred_norm
                
                if 'subtitles' in dic:
                    del dic['subtitles']
                
                self.experience_pool.append({
                    'question': dic.get('question', ''),
                    'gt_answer': gt_norm,
                    'pred_answer': pred_norm,
                    'score': score,
                    'analyzed_videos': list(analyzed_videos - failed_videos),
                    'video_descriptions': dict(all_tool_results),
                    'failed_videos': list(failed_videos),
                    'num_turns': cur_turn,
                })
                return {
                    'messages': messages,
                    'raw_data': dic,
                    'score': score,
                    'video_descriptions': dict(all_tool_results),
                }

            # 双重校验：除 analyzed_videos 外，再看 all_tool_results 是否真的有内容。
            pending_by_evidence = {
                vid
                for vid in (required_videos - failed_videos)
                if not (all_tool_results.get(vid) or "").strip()
            }
            pending_videos = pending_videos | pending_by_evidence
            sim_ready, sim_margin, sim_top_vid, sim_scores = self._similarity_rank_signal(
                dic, required_videos - failed_videos, tool_results
            )
            if self._is_similarity_centric_question(dic) and sim_ready:
                pending_videos = set()

            if pending_videos:
                self._append_message(messages, "user",
                    f"You haven't read video(s) {sorted(pending_videos)} yet. "
                    f"Read them before answering.")
            else:
                # 所有视频已读，检查证据是否充分
                _unclear = self._find_uncertain_videos_in_tool_result(
                    "\n".join(all_tool_results.get(v, "") for v in video_ids),
                    video_ids,
                )
                if _unclear and cur_turn < MAX_DS_ROUND - 1:
                    self._append_message(messages, "user",
                        f"All videos read but evidence is unclear for {_unclear}. You can:\n"
                        f"1. Re-read with a more specific question\n"
                        f"2. Use <scene_graph id=\"VID\">0:end;prompts=OBJECT;model=groundingdino</scene_graph> to detect/count objects\n"
                        f"3. Use <visual_similarity id=\"A,B\">a=0:10;b=0:10;model=clip</visual_similarity> to compare videos\n"
                        f"4. Output your answer inside <answer></answer> tags")
                else:
                    prompt_version_now = str(getattr(self.args, "prompt_version", "default") or "default").lower()
                    if prompt_version_now == "short":
                        self._append_message(
                            messages,
                            "user",
                            "All required videos have valid evidence. "
                            "Now output ONLY final <answer>X</answer>. "
                            "Do not call tools.",
                        )
                    else:
                        self._append_message(messages, "user",
                            "Output your answer inside <answer></answer> tags.")

        # ================================================================
        # Gate C: Agent 失败兜底 — 用 direct VLM 缓存或重新调用
        # ================================================================
        # short 严格模式：在任何 fallback 前，先确保剩余视频已读
        pending_videos = required_videos - analyzed_videos - failed_videos
        pending_by_evidence = {
            vid
            for vid in (required_videos - failed_videos)
            if not (all_tool_results.get(vid) or "").strip()
        }
        pending_videos = pending_videos | pending_by_evidence
        sim_ready, sim_margin, sim_top_vid, sim_scores = self._similarity_rank_signal(
            dic, required_videos - failed_videos, tool_results
        )
        if self._is_similarity_centric_question(dic) and sim_ready:
            pending_videos = set()
        is_short = str(getattr(self.args, "prompt_version", "default")).lower() == "short"
        short_require_all = os.getenv("SHORT_REQUIRE_ALL_READ", "1") == "1"
        if is_short and short_require_all and pending_videos:
            logger.warning(f"[SHORT_STRICT_READ] pending before fallback: {sorted(pending_videos)}; auto-read pending videos")
            auto_calls = []
            for vid in sorted(pending_videos):
                dur = float(dic.get('duration_map', {}).get(vid, 10) or 10)
                auto_calls.append(
                    f'<video_reader id="{vid}">0:{dur:.1f}</video_reader>'
                    f'<video_reader_question>What is the main action and key object?</video_reader_question>'
                )
            auto_text = "\n".join(auto_calls)
            auto_result = self.process_video_reader(auto_text, dic, all_tool_results)
            if auto_result:
                self._append_tool_result(tool_results, "video_reader", auto_result)
                self._append_message(messages, "user", auto_result)
            for vid in list(pending_videos):
                if all_tool_results.get(vid):
                    analyzed_videos.add(vid)
            pending_videos = required_videos - analyzed_videos - failed_videos

        if is_short and short_require_all and pending_videos:
            logger.warning(f"[SHORT_STRICT_READ] still pending {sorted(pending_videos)}; fallback to direct VLM")
            direct_fb = self._direct_vlm_answer(dic, answer_choices)
            if direct_fb and direct_fb in answer_set:
                pred_norm, gt_norm, score = self._score_prediction(dic, direct_fb, answer_choices)
                dic['pred_answer'] = pred_norm
                logger.info(
                    f"[DIRECT_VLM_FALLBACK] short strict pending -> direct answer: {pred_norm} "
                    f"(gt={gt_norm}, correct={score})"
                )
                if 'subtitles' in dic:
                    del dic['subtitles']
                return {
                    'messages': messages,
                    'raw_data': dic,
                    'score': score,
                    'video_descriptions': dict(all_tool_results),
                }
            logger.warning("[DIRECT_VLM_FALLBACK] short strict pending -> direct VLM failed, continue generic fallback")

        # short 严格模式：都读完后优先强制一次最终答案，避免“证据齐全但空收尾”
        if is_short and short_require_all and not pending_videos:
            final_try = self._force_answer(messages, answer_choices)
            if final_try:
                answer = self.extract_final_answer(final_try, allowed_choices=answer_set)
                if answer in answer_set:
                    pred_norm, gt_norm, score = self._score_prediction(dic, answer, answer_choices)
                    dic['pred_answer'] = pred_norm
                    logger.info(
                        f"[SHORT_FINAL_FORCE] all videos read, forced final answer: {pred_norm} "
                        f"(gt={gt_norm}, correct={score})"
                    )
                    if 'subtitles' in dic:
                        del dic['subtitles']
                    return {
                        'messages': messages,
                        'raw_data': dic,
                        'score': score,
                        'video_descriptions': dict(all_tool_results),
                    }

        if direct_threshold > 0:
            fallback = _direct_answer_cache  # 优先用缓存（已经跑过一次）
            if not fallback or fallback not in answer_set:
                # 缓存没结果，重新跑一次（可能之前 n_videos <= threshold 时成功
                # 但选择跑 agent，现在 agent 也失败了）
                fallback = self._direct_vlm_answer(dic, answer_choices)
            if fallback and fallback in answer_set:
                pred_norm, gt_norm, score = self._score_prediction(dic, fallback, answer_choices)
                dic['pred_answer'] = pred_norm
                logger.info(
                    f"[DIRECT_VLM_FALLBACK] Agent failed, using direct answer: {pred_norm} "
                    f"(gt={gt_norm}, correct={score})"
                )
                if 'subtitles' in dic:
                    del dic['subtitles']
                return {
                    'messages': messages,
                    'raw_data': dic,
                    'score': score,
                    'video_descriptions': dict(all_tool_results),
                }

        # ================================================================
        # Gate D: 解析为空兜底 — 强制选择最可能答案；仍失败则随机
        # ================================================================
        prompt_version = str(getattr(self.args, "prompt_version", "default") or "default").lower()
        if prompt_version == "short":
            short_use_direct_on_empty = os.getenv("SHORT_DIRECT_ON_EMPTY_FALLBACK", "1") == "1"
        else:
            short_use_direct_on_empty = False

        if short_use_direct_on_empty:
            direct_empty_fb = self._direct_vlm_answer(dic, answer_choices)
            if direct_empty_fb and direct_empty_fb in answer_set:
                pred_norm, gt_norm, score = self._score_prediction(dic, direct_empty_fb, answer_choices)
                dic['pred_answer'] = pred_norm
                logger.info(
                    f"[DIRECT_VLM_EMPTY_FALLBACK] Parsed empty, using direct answer: {pred_norm} "
                    f"(gt={gt_norm}, correct={score})"
                )
                if 'subtitles' in dic:
                    del dic['subtitles']
                return {
                    'messages': messages,
                    'raw_data': dic,
                    'score': score,
                    'video_descriptions': dict(all_tool_results),
                }

        force_answer = self._force_answer_from_evidence(dic, all_tool_results, answer_choices)
        if force_answer and force_answer in answer_set:
            pred_norm, gt_norm, score = self._score_prediction(dic, force_answer, answer_choices)
            dic['pred_answer'] = pred_norm
            logger.info(
                f"[FORCE_ANSWER_FALLBACK] Parsed empty, forcing answer: {pred_norm} "
                f"(gt={gt_norm}, correct={score})"
            )
            if 'subtitles' in dic:
                del dic['subtitles']
            return {
                'messages': messages,
                'raw_data': dic,
                'score': score,
                'video_descriptions': dict(all_tool_results),
            }

        if answer_choices:
            rand_answer = random.choice(answer_choices)
            pred_norm, gt_norm, score = self._score_prediction(dic, rand_answer, answer_choices)
            dic['pred_answer'] = pred_norm
            logger.warning(
                f"[RANDOM_FALLBACK] Parsed empty and force failed, random answer: {pred_norm} "
                f"(gt={gt_norm}, correct={score})"
            )
            if 'subtitles' in dic:
                del dic['subtitles']
            return {
                'messages': messages,
                'raw_data': dic,
                'score': score,
                'video_descriptions': dict(all_tool_results),
            }

        if 'subtitles' in dic:
            del dic['subtitles']
        return {'messages': messages, 'raw_data': dic, 'score': False}

    # =========================================================================
    # 工具处理
    # =========================================================================

    def process_tool_calls(
        self,
        output_text: str,
        dic: Dict,
        tool_cache: Optional[Dict[str, str]] = None,
        tool_results: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """分发工具调用"""
        prompt_version = str(getattr(self.args, "prompt_version", "default") or "default").lower()
        if prompt_version == "short":
            output_text = self._strip_think_content(output_text)
        else:
            pass
        if not output_text:
            return ""
        tool_result = ""

        if "<temporal_grounding_agent" in output_text:
            for vid_id, query in RE_TG_AGENT.findall(output_text):
                if vid_id in dic.get("video_map", {}):
                    result, dic = self.process_temporal_grounding(query.strip(), vid_id.strip(), dic)
                    tool_result += f"{result}\n"
                    self._append_tool_result(tool_results, "temporal_grounding_agent", result)

        if "<video_reader" in output_text and "<video_reader_question>" in output_text:
            vr = self.process_video_reader(output_text, dic, tool_cache)
            tool_result += vr
            self._append_tool_result(tool_results, "video_reader", vr)
            # Detection 工具（scene_graph, object_tracker）不再自动追加。
            # Planner 可自行调用；系统在证据不足时会提示可用工具。

        if "<video_segment_retriever_textual_query" in output_text:
            for vid_id, query in RE_SEGMENT_QUERY.findall(output_text):
                if dic.get("searchable_map", {}).get(vid_id.strip(), False):
                    res = self.process_video_segment_retriever_text(vid_id.strip(), query.strip(), dic)
                    tool_result += res
                    self._append_tool_result(tool_results, "video_segment_retriever", res)
                else:
                    res = f"Video {vid_id}: No index. Use video_reader.\n"
                    tool_result += res
                    self._append_tool_result(tool_results, "video_segment_retriever", res)

        if "<video_browser" in output_text:
            # video_browser 已废弃，引导模型使用 video_reader
            tool_result += "video_browser is deprecated. Use video_reader instead for more accurate results.\n"

        if "<subtitle_retriever" in output_text:
            sr = self.process_subtitle_retriever(output_text, dic)
            tool_result += sr
            self._append_tool_result(tool_results, "subtitle_retriever", sr)

        if "<subtitle_extractor" in output_text:
            se = self.process_subtitle_extractor(output_text, dic)
            tool_result += se
            self._append_tool_result(tool_results, "subtitle_extractor", se)

        if "<scene_detector" in output_text:
            sd = self.process_scene_detector(output_text, dic)
            tool_result += sd
            self._append_tool_result(tool_results, "scene_detector", sd)

        if "<object_tracker" in output_text:
            ot = self.process_object_tracker(output_text, dic)
            tool_result += ot
            self._append_tool_result(tool_results, "object_tracker", ot)

        if "<spatial_relation" in output_text:
            sr = self.process_spatial_relation(output_text, dic)
            tool_result += sr
            self._append_tool_result(tool_results, "spatial_relation", sr)

        if "<scene_graph" in output_text:
            sg = self.process_scene_graph(output_text, dic)
            tool_result += sg
            self._append_tool_result(tool_results, "scene_graph", sg)

        if "<visual_similarity" in output_text:
            vs = self.process_visual_similarity(output_text, dic)
            tool_result += vs
            self._append_tool_result(tool_results, "visual_similarity", vs)

        return tool_result

    def process_video_reader(self, output_text: str, dic: Dict, tool_cache: Optional[Dict[str, str]] = None) -> str:
        """处理视频阅读请求"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        # 兼容模型输出的错误闭合标签
        if "<video_reader_question>" in output_text and "</video_reader_question>" not in output_text:
            output_text = re.sub(
                r"(<video_reader_question>.*?)(</video_reader>)",
                r"\1</video_reader_question>",
                output_text,
                flags=re.S,
            )

        matches = RE_VIDEO_READER.findall(output_text.strip())
        if not matches:
            return ""

        # Counting 任务：为 video_reader query 追加计数指令，让 VLM 明确返回数字
        _task = dic.get("task", "")
        if isinstance(_task, list):
            _task = _task[0] if _task else ""
        _is_counting = bool(_task == "Counting" or re.search(
            r'\b(how\s+many|count|number\s+of|most\s+\w+)\b',
            dic.get("question", "").lower(),
        ))

        def _parse_time_match(tm_raw: str, fallback_duration: Optional[float] = None):
            tm = str(tm_raw).strip()
            tm = (
                tm.replace("–", "-")
                  .replace("—", "-")
            )
            tm = re.sub(r"\s+to\s+", "-", tm, flags=re.I)
            tm = tm.strip().rstrip(";,.")
            default_end = float(fallback_duration if fallback_duration is not None else (dic.get("duration", 0) or 0))

            def _hms_to_seconds(hms_text: str) -> float:
                # 支持 HH:MM:SS(.mmm) 与 MM:SS(.mmm)
                s = hms_text.strip()
                parts = s.split(":")
                if len(parts) == 3:
                    h, m, sec = parts
                    return float(h) * 3600.0 + float(m) * 60.0 + float(sec)
                if len(parts) == 2:
                    m, sec = parts
                    return float(m) * 60.0 + float(sec)
                return float(s)

            def _parse_time_token(token: str) -> Optional[float]:
                s = str(token).strip().strip("()[]{}<>\"'").rstrip(";,.")
                if not s:
                    return None
                if re.fullmatch(r"\d+:\d{1,2}:\d{1,2}(?:\.\d+)?", s):
                    return _hms_to_seconds(s)
                if re.fullmatch(r"\d+:\d{1,2}(?:\.\d+)?", s):
                    return _hms_to_seconds(s)
                if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s):
                    return float(s)
                return None

            begin, end = None, None

            # 1) 显式区间：a-b（支持 HH:MM:SS / MM:SS / seconds 混合）
            if "-" in tm:
                segs = re.split(r"\s*-\s*", tm, maxsplit=1)
                if len(segs) == 2:
                    b_val = _parse_time_token(segs[0])
                    e_val = _parse_time_token(segs[1])
                    if b_val is not None and e_val is not None:
                        begin, end = float(b_val), float(e_val)

            # 2) 单个 HH:MM:SS（如 0:00:10）按 [0, t] 解释
            if begin is None or end is None:
                if re.fullmatch(r"\d+:\d{1,2}:\d{1,2}(?:\.\d+)?", tm):
                    begin, end = 0.0, float(_hms_to_seconds(tm))

            # 3) 简单 a:b（秒区间）
            if begin is None or end is None:
                m_simple = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", tm)
                if m_simple:
                    begin, end = float(m_simple.group(1)), float(m_simple.group(2))

            # 4) 单值（seconds 或 MM:SS）按 [0, t] 解释
            if begin is None or end is None:
                single = _parse_time_token(tm)
                if single is not None:
                    begin, end = 0.0, float(single)

            # 5) 保底：交给历史解析器
            if begin is None or end is None:
                begin, end = parse_time_range({"range": tm}, 0.0, default_end)

            # 按当前视频时长裁剪，避免越界导致空片段
            if default_end > 0:
                begin = max(0.0, min(float(begin), default_end))
                end = max(0.0, min(float(end), default_end))

            # 兜底：若仍非法区间，回退到整段（当前视频），防止 empty segment 链式失败
            if end <= begin:
                begin, end = 0.0, default_end
            return begin, end

        # Resolve vid_id early so we can coalesce ranges per video when needed.
        resolved_matches = []
        for vid_id, time_match, query in matches:
            vid_id = self._resolve_video_id(vid_id, dic)
            resolved_matches.append((vid_id, time_match, query))

        # Counting: avoid dense 10s stepping. Keep only a few coarse segments per video.
        if _is_counting:
            max_seg = float(getattr(self.args, "counting_max_seconds", 0.0))
            max_seg_count = int(getattr(self.args, "counting_max_segments", 1))
            # Make each counted window coarser to reduce churn.
            # Bigger window does NOT necessarily increase cost because we cap max frames.
            # Default policy: minimum window length = video_duration / 5
            # (bounded by [1s, duration] and an optional max cap).
            min_win_frac = float(os.getenv("COUNTING_MIN_WINDOW_FRAC", "0.2"))  # 1/5 by default
            max_win = float(os.getenv("COUNTING_MAX_WINDOW_SECONDS", "60"))
            from collections import defaultdict
            seen = defaultdict(int)
            filtered = []
            for vid_id, time_match, query in resolved_matches:
                if seen[vid_id] >= max_seg_count:
                    continue
                duration = float(dic.get('duration_map', {}).get(vid_id, dic.get("duration", 0) or 0))
                begin, end = _parse_time_match(time_match, fallback_duration=duration)
                if end <= begin:
                    # If model gives an invalid range, fall back to a short window.
                    end = begin + (max_seg if max_seg > 0 else 1.0)
                # Enforce a dynamic minimum window length to avoid overly dense 0:12 style chopping.
                # - short videos: read the whole video
                # - long videos: use a coarser window (bounded by max_win)
                if duration and duration > 0:
                    dyn_min_win = float(duration) * float(min_win_frac)
                    dyn_min_win = max(1.0, min(float(duration), dyn_min_win))
                    if max_win > 0:
                        dyn_min_win = min(float(max_win), dyn_min_win)
                else:
                    dyn_min_win = 1.0
                if duration and dyn_min_win > 0 and (end - begin) < dyn_min_win:
                    target_end = min(float(duration), float(begin) + dyn_min_win)
                    # If begin is too close to the end, shift left.
                    if target_end - begin < dyn_min_win:
                        begin = max(0.0, target_end - dyn_min_win)
                    end = target_end
                if max_seg > 0 and end - begin > max_seg:
                    end = begin + max_seg
                seen[vid_id] += 1
                filtered.append((vid_id, f"{begin:.1f}:{end:.1f}", query))
            matches = filtered
        else:
            matches = resolved_matches

        tasks = []
        error_result = ''
        vid_ids, time_matches = [], []
        seen_reads = set()
        
        for vid_id, time_match, query in matches:
            video_path = dic['video_map'].get(vid_id, dic.get('video_path'))
            if not video_path:
                continue
            duration = float(dic.get('duration_map', {}).get(vid_id, dic.get("duration", 0) or 0))

            # short 模式默认直接读整段视频，避免时间段格式噪声导致 empty segment。
            force_full_read = (str(getattr(self.args, "prompt_version", "default")).lower() == "short")
            force_full_read = force_full_read and (os.getenv("SHORT_VIDEO_READER_FULL", "1") == "1")
            if force_full_read and duration > 0:
                begin, end = 0.0, float(duration)
                time_match = f"{begin:.1f}:{end:.1f}"
            else:
                begin, end = _parse_time_match(time_match, fallback_duration=duration)

            # Counting 任务：在 query 末尾追加明确计数要求（仅当 query 本身未包含计数关键词时）
            if _is_counting and not re.search(r'\bcount\b|\bhow many\b|\bnumber\b', query.lower()):
                query = query.rstrip('. ') + ". Count the exact number of relevant objects and report as a specific integer."
            # Skip duplicate reads for the same video/time/query.
            q_norm = re.sub(r"\s+", " ", str(query)).strip().lower()
            key = (vid_id, float(begin), float(end), q_norm)
            if key in seen_reads:
                continue
            seen_reads.add(key)
            
            # VLM prompt：只保留 VLM 擅长的字段（场景/动作/空间/证据）。
            # 物体检测交给 scene_graph / object_tracker（GroundingDINO），
            # 避免 VLM 在 OBJECTS 字段产生 "1 car, 1 car, ..." 无限重复幻觉。
            _VR_PROMPT_TEMPLATE = (
                "{prefix}"
                "Answer the question with a structured report.\n"
                "Question: {query}\n\n"
                "Output format:\n"
                "CAPTION: <1-2 sentences: scene, setting, main activity>\n"
                "ACTIONS: <what is happening, temporal order if multiple events>\n"
                "SPATIAL: <positions and spatial relationships, e.g. person LEFT of car>\n\n"
                "EVIDENCE: <1-2 sentences of specific visual facts MOST relevant to the question>\n\n"
                "Rules: Be specific. "
                "Describe what you SEE, not what you guess. "
                "Do NOT list objects one by one. "
                "Do NOT choose among options. Do NOT output A/B/C/D."
            )

            _vp = str(video_path).lower()
            if _vp.endswith((".jpg", ".jpeg", ".png", ".webp")) and os.path.exists(video_path):
                # Single-frame input: bypass timestamp slicing and feed the image directly.
                video_clip, timestamps = [video_path], [0.0]
                if getattr(self.args, "video_reader_caption", 1):
                    query_formatted = _VR_PROMPT_TEMPLATE.format(
                        prefix=f"[Video {vid_id}] Inspect the image carefully. ",
                        query=query,
                    )
                else:
                    query_formatted = f"[Video {vid_id}] {query}"
                time_match = "image"
            elif 'haystack' in video_path:
                video_clip, timestamps = timestamp_to_frames(video_path, begin, end, f'{self.full_dataset_folder}/dense_frames')
                if _is_counting:
                    max_vr_frames = int(getattr(self.args, "counting_video_reader_max_frames", 8))
                    if max_vr_frames > 0 and len(video_clip) > max_vr_frames:
                        step = max(1, len(video_clip) // max_vr_frames)
                        video_clip = video_clip[::step][:max_vr_frames]
                        timestamps = timestamps[::step][:max_vr_frames]
                if getattr(self.args, "video_reader_caption", 1):
                    query_formatted = _VR_PROMPT_TEMPLATE.format(
                        prefix=f"[Video {vid_id}] Analyze the frames carefully. ",
                        query=query,
                    )
                else:
                    query_formatted = f"[Video {vid_id}] Analyze: {query}"
            else:
                vr_fps = self.args.clip_fps
                if _is_counting:
                    vr_fps = float(getattr(self.args, "counting_video_reader_fps", vr_fps))
                video_clip, timestamps = timestamp_to_clip_path(
                    self.full_dataset_folder, begin, end, video_path, fps=vr_fps
                )
                # If extraction failed for this window, retry once with a safer, wider window.
                # This avoids cascaded "empty segment" when the planner picks a window with no frames.
                if (not video_clip) and duration and float(duration) > 0 and (float(begin) > 0.0 or float(end) < float(duration)):
                    retry_begin, retry_end = 0.0, float(duration)
                    retry_fps = max(1.0, float(vr_fps))
                    video_clip, timestamps = timestamp_to_clip_path(
                        self.full_dataset_folder, retry_begin, retry_end, video_path, fps=retry_fps
                    )
                if _is_counting:
                    max_vr_frames = int(getattr(self.args, "counting_video_reader_max_frames", 8))
                    if max_vr_frames > 0 and len(video_clip) > max_vr_frames:
                        step = max(1, len(video_clip) // max_vr_frames)
                        video_clip = video_clip[::step][:max_vr_frames]
                        timestamps = timestamps[::step][:max_vr_frames]
                if getattr(self.args, "video_reader_caption", 1):
                    query_formatted = _VR_PROMPT_TEMPLATE.format(
                        prefix=f"[Video {vid_id}] Watch the clip carefully. ",
                        query=query,
                    )
                else:
                    query_formatted = f"[Video {vid_id}] {query}"
            
            if video_clip:
                tasks.append((query_formatted, video_clip, timestamps))
                vid_ids.append(vid_id)
                time_matches.append(time_match)
            else:
                error_result += f"Video {vid_id} at {time_match}: empty segment.\n"

        if not tasks:
            return error_result

        results = self.batch_video2text(tasks)
        
        ans = error_result
        for i, result in enumerate(results):
            vid_id = vid_ids[i]
            if getattr(self.args, "video_reader_caption", 1):
                caption, evidence = _split_caption_evidence(result)
                merged = ""
                if caption:
                    merged += f"CAPTION: {caption}\n"
                merged += f"EVIDENCE: {evidence}" if evidence else (f"EVIDENCE: {result}" if result else "EVIDENCE: ")
                ans += f'<video_reader id="{vid_id}">{time_matches[i]}</video_reader> result: {merged}\n'
                if tool_cache is not None and merged.strip():
                    tool_cache[vid_id] = (tool_cache.get(vid_id, "") + "\n" + merged).strip()
            else:
                ans += f'<video_reader id="{vid_id}">{time_matches[i]}</video_reader> result: {result}\n'
                if tool_cache is not None and result:
                    tool_cache[vid_id] = (tool_cache.get(vid_id, "") + "\n" + result).strip()
        
        return ans

    def process_video_browser(self, output_text: str, dic: Dict, tool_cache: Optional[Dict[str, str]] = None) -> str:
        """处理视频浏览请求"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_VIDEO_BROWSER.findall(output_text.strip())
        if not matches:
            return ""
        
        results = []
        for vid_id, query in matches:
            vid_id = self._resolve_video_id(vid_id, dic)
            if vid_id not in dic['video_map']:
                results.append(f"Video {vid_id}: not found.\n")
                continue
            
            video_path = dic['video_map'][vid_id]
            duration = dic['duration_map'][vid_id]
            
            if 'haystack' in video_path:
                video_clip, timestamps = timestamp_to_frames(video_path, 0, duration, f'{self.full_dataset_folder}/dense_frames')
            else:
                video_clip, timestamps = timestamp_to_clip_path(self.full_dataset_folder, 0, duration, video_path, fps=self.args.clip_fps)
            
            if video_clip:
                ans = self.batch_video2text([(query, video_clip, timestamps)])[0]
                results.append(f"<video_browser id=\"{vid_id}\">{query}</video_browser> result: {ans}\n")
                if tool_cache is not None and ans:
                    tool_cache[vid_id] = (tool_cache.get(vid_id, "") + "\n" + ans).strip()
            else:
                results.append(f"Video {vid_id}: failed to read.\n")
        
        return "".join(results)

    def process_scene_detector(self, output_text: str, dic: Dict) -> str:
        """处理场景切换检测请求"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_SCENE_DETECTOR.findall(output_text.strip())
        if not matches:
            return ""

        results = []
        for vid_id, query in matches:
            if not getattr(self.args, "enable_scene_detector", 1):
                results.append(
                    f'<scene_detector id="{vid_id}">{query}</scene_detector> result: disabled\n'
                )
                continue
            vid_id = self._resolve_video_id(vid_id, dic)
            if vid_id not in dic.get('video_map', {}):
                results.append(f"Video {vid_id}: not found.\n")
                continue
            params = parse_tool_params(query)
            try:
                fps = float(params.get("fps", 1.0))
                threshold = float(params.get("threshold", 0.6))
                bins = int(params.get("bins", 32))
                min_gap = float(params.get("min_gap", 0.5))
            except ValueError:
                fps, threshold, bins, min_gap = 1.0, 0.6, 32, 0.5

            video_path = dic['video_map'][vid_id]
            changes = detect_scene_changes(
                video_path,
                sample_fps=fps,
                hist_bins=bins,
                threshold=threshold,
                min_gap=min_gap,
            )
            results.append(
                f'<scene_detector id="{vid_id}">{query}</scene_detector> result: {changes}\n'
            )
        return "".join(results)

    def process_object_tracker(self, output_text: str, dic: Dict) -> str:
        """处理目标检测/跟踪请求"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_OBJECT_TRACKER.findall(output_text.strip())
        if not matches:
            return ""

        results = []
        for vid_id, query in matches:
            if not getattr(self.args, "enable_object_tracker", 1):
                results.append(
                    f'<object_tracker id="{vid_id}">{query}</object_tracker> result: disabled\n'
                )
                continue
            vid_id = self._resolve_video_id(vid_id, dic)
            if vid_id not in dic.get('video_map', {}):
                results.append(f"Video {vid_id}: not found.\n")
                continue
            params = parse_tool_params(query)
            duration = dic['duration_map'].get(vid_id, DEFAULT_VIDEO_DURATION)
            begin, end = parse_time_range(params, 0.0, float(duration))
            fps = float(params.get("fps", self.args.clip_fps))
            target = params.get("target")
            model_name = params.get("model", "yolov8m.pt")
            conf = float(params.get("conf", 0.25))

            video_path = dic['video_map'][vid_id]
            cache_key = (video_path, float(begin), float(end), float(fps))
            cached = self._pretool_frame_cache.get(cache_key)
            if cached:
                frame_paths, timestamps = cached
            else:
                frame_paths, timestamps = timestamp_to_clip_path(
                    self.full_dataset_folder, begin, end, video_path, fps=fps
                )
            det_results = run_object_detection(
                frame_paths,
                timestamps,
                target=target,
                model_name=model_name,
                conf=conf,
            )
            summary = summarize_object_detection(det_results, frame_paths=frame_paths, timestamps=timestamps)
            results.append(
                f'<object_tracker id="{vid_id}"> result:\n{summary}\n'
            )
        return "".join(results)

    def process_spatial_relation(self, output_text: str, dic: Dict) -> str:
        """处理空间关系抽取请求"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_SPATIAL_RELATION.findall(output_text.strip())
        if not matches:
            return ""

        results = []
        for vid_id, query in matches:
            if not getattr(self.args, "enable_spatial_relation", 1):
                results.append(
                    f'<spatial_relation id="{vid_id}">{query}</spatial_relation> result: disabled\n'
                )
                continue
            vid_id = self._resolve_video_id(vid_id, dic)
            if vid_id not in dic.get('video_map', {}):
                results.append(f"Video {vid_id}: not found.\n")
                continue
            params = parse_tool_params(query)
            duration = dic['duration_map'].get(vid_id, DEFAULT_VIDEO_DURATION)
            begin, end = parse_time_range(params, 0.0, float(duration))
            fps = float(params.get("fps", self.args.clip_fps))
            targets = params.get("targets") or params.get("target")
            model_name = params.get("model", "yolov8n.pt")
            conf = float(params.get("conf", 0.25))

            video_path = dic['video_map'][vid_id]
            cache_key = (video_path, float(begin), float(end), float(fps))
            cached = self._pretool_frame_cache.get(cache_key)
            if cached:
                frame_paths, timestamps = cached
            else:
                frame_paths, timestamps = timestamp_to_clip_path(
                    self.full_dataset_folder, begin, end, video_path, fps=fps
                )
            relation_results = run_spatial_relations(
                frame_paths,
                timestamps,
                targets=targets,
                model_name=model_name,
                conf=conf,
            )
            summary = summarize_spatial_relations(relation_results)
            results.append(
                f'<spatial_relation id="{vid_id}"> result:\n{summary}\n'
            )
        return "".join(results)

    def process_scene_graph(self, output_text: str, dic: Dict) -> str:
        """处理场景图抽取请求"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_SCENE_GRAPH.findall(output_text.strip())
        if not matches:
            return ""

        results = []
        for vid_id, query in matches:
            if not getattr(self.args, "enable_scene_graph", 1):
                results.append(
                    f'<scene_graph id="{vid_id}">{query}</scene_graph> result: disabled\n'
                )
                continue
            vid_id = self._resolve_video_id(vid_id, dic)
            if vid_id not in dic.get('video_map', {}):
                results.append(f"Video {vid_id}: not found.\n")
                continue
            params = parse_tool_params(query)
            model_type = (params.get("model") or self.args.scene_graph_model).lower()
            duration = dic['duration_map'].get(vid_id, DEFAULT_VIDEO_DURATION)
            begin, end = parse_time_range(params, 0.0, float(duration))
            fps = float(params.get("fps", self.args.clip_fps))
            prompts = params.get("prompts") or params.get("targets") or params.get("target")
            model_name = params.get("model_name")
            conf = float(params.get("conf", 0.25))
            max_objects = int(params.get("max_objects", self.args.scene_graph_max_objects))
            near_thresh = float(params.get("near_thresh", 0.25))
            max_frames = int(params.get("max_frames", self.args.scene_graph_max_frames))
            _task = dic.get("task", "")
            if isinstance(_task, list):
                _task = _task[0] if _task else ""
            if _task == "Counting":
                # Counting: sample only a few frames to keep cost low.
                max_frames = min(max_frames, int(getattr(self.args, "counting_scene_graph_max_frames", 12)))

            video_path = dic['video_map'][vid_id]
            cache_key = (video_path, float(begin), float(end), float(fps))
            cached = self._pretool_frame_cache.get(cache_key)
            if cached:
                frame_paths, timestamps = cached
            else:
                frame_paths, timestamps = timestamp_to_clip_path(
                    self.full_dataset_folder, begin, end, video_path, fps=fps
                )
            if model_type in ("groundingdino", "dino", "gdino"):
                graph_results = run_scene_graph_grounded(
                    frame_paths,
                    timestamps,
                    prompts=prompts,
                    model_name=model_name or self.args.groundingdino_model,
                    device=self.args.scene_graph_device,
                    box_threshold=float(params.get("box_threshold", self.args.groundingdino_box_threshold)),
                    text_threshold=float(params.get("text_threshold", self.args.groundingdino_text_threshold)),
                    max_frames=max_frames,
                    max_objects=max_objects,
                    near_thresh=near_thresh,
                )
            elif model_type in ("yolo", "yolov8", "detector"):
                graph_results = run_scene_graph(
                    frame_paths,
                    timestamps,
                    targets=prompts,
                    model_name=model_name or "yolov8n.pt",
                    conf=conf,
                    max_frames=max_frames,
                    max_objects=max_objects,
                    near_thresh=near_thresh,
                )
            else:
                graph_results = [{"error": f"unknown_scene_graph_model:{model_type}"}]
            # 将原始 JSON 结构转为紧凑的自然语言摘要，大幅节省 token
            summary = summarize_scene_graph_results(graph_results)
            results.append(
                f'<scene_graph id="{vid_id}"> result:\n{summary}\n'
            )
        return "".join(results)

    def process_visual_similarity(self, output_text: str, dic: Dict) -> str:
        """处理视觉相似度请求"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_VISUAL_SIMILARITY.findall(output_text.strip())
        if not matches:
            return ""

        results = []
        for vid_id, query in matches:
            if not getattr(self.args, "enable_visual_similarity", 1):
                results.append(
                    f'<visual_similarity id="{vid_id}">{query}</visual_similarity> result: disabled\n'
                )
                continue
            vid_ids = [v.strip() for v in re.split(r"[,\s]+", vid_id) if v.strip()]
            if len(vid_ids) != 2:
                results.append(f"visual_similarity: invalid id '{vid_id}'.\n")
                continue
            vid_a = self._resolve_video_id(vid_ids[0], dic)
            vid_b = self._resolve_video_id(vid_ids[1], dic)
            if vid_a not in dic.get('video_map', {}) or vid_b not in dic.get('video_map', {}):
                results.append(f"visual_similarity: video not found for '{vid_id}'.\n")
                continue

            params = parse_tool_params(query)
            model_type = (params.get("model") or self.args.visual_similarity_model).lower()
            device = params.get("device", self.args.visual_similarity_device)
            range_a = params.get("a") or params.get("a_range")
            range_b = params.get("b") or params.get("b_range")
            if (not range_a or not range_b) and params.get("_positional"):
                pos = params["_positional"]
                if len(pos) >= 2:
                    range_a = range_a or pos[0]
                    range_b = range_b or pos[1]

            duration_a = dic['duration_map'].get(vid_a, DEFAULT_VIDEO_DURATION)
            duration_b = dic['duration_map'].get(vid_b, DEFAULT_VIDEO_DURATION)
            begin_a, end_a = parse_time_range({"range": range_a}, 0.0, float(duration_a))
            begin_b, end_b = parse_time_range({"range": range_b}, 0.0, float(duration_b))
            fps = float(params.get("fps", self.args.clip_fps))
            bins = int(params.get("bins", 32))
            max_frames = int(params.get("max_frames", self.args.visual_similarity_max_frames))

            video_path_a = dic['video_map'][vid_a]
            video_path_b = dic['video_map'][vid_b]
            cache_a = self._pretool_frame_cache.get((video_path_a, float(begin_a), float(end_a), float(fps)))
            cache_b = self._pretool_frame_cache.get((video_path_b, float(begin_b), float(end_b), float(fps)))
            if cache_a:
                frames_a, _ = cache_a
            else:
                frames_a, _ = timestamp_to_clip_path(
                    self.full_dataset_folder, begin_a, end_a, video_path_a, fps=fps
                )
            if cache_b:
                frames_b, _ = cache_b
            else:
                frames_b, _ = timestamp_to_clip_path(
                    self.full_dataset_folder, begin_b, end_b, video_path_b, fps=fps
                )
            if model_type in ("color_hist", "hist"):
                score, method = visual_similarity_from_frames(frames_a, frames_b, bins=bins)
            elif model_type == "clip":
                emb_a = _image_embeddings(
                    frames_a,
                    "clip",
                    self.args.visual_similarity_clip_model,
                    device,
                    max_frames,
                )
                emb_b = _image_embeddings(
                    frames_b,
                    "clip",
                    self.args.visual_similarity_clip_model,
                    device,
                    max_frames,
                )
                if emb_a is None or emb_b is None:
                    results.append(
                        f'<visual_similarity id="{vid_id}">{query}</visual_similarity> result: '
                        '{"error": "clip_unavailable"}\n'
                    )
                    continue
                score = visual_similarity_from_embeddings(emb_a, emb_b)
                method = "clip"
            elif model_type == "dinov2":
                emb_a = _image_embeddings(
                    frames_a,
                    "dinov2",
                    self.args.visual_similarity_dinov2_model,
                    device,
                    max_frames,
                )
                emb_b = _image_embeddings(
                    frames_b,
                    "dinov2",
                    self.args.visual_similarity_dinov2_model,
                    device,
                    max_frames,
                )
                if emb_a is None or emb_b is None:
                    results.append(
                        f'<visual_similarity id="{vid_id}">{query}</visual_similarity> result: '
                        '{"error": "dinov2_unavailable"}\n'
                    )
                    continue
                score = visual_similarity_from_embeddings(emb_a, emb_b)
                method = "dinov2"
            else:
                results.append(
                    f'<visual_similarity id="{vid_id}">{query}</visual_similarity> result: '
                    f'{{"error": "unknown_model:{model_type}"}}\n'
                )
                continue
            results.append((vid_a, vid_b, round(score, 4), method))

        # 将所有配对压缩成一行紧凑输出
        if not results:
            return ""
        # results 现在混合了字符串(error) 和 tuple(成功)
        error_lines = [r for r in results if isinstance(r, str)]
        score_tuples = [r for r in results if isinstance(r, tuple)]
        parts: List[str] = []
        if score_tuples:
            method_used = score_tuples[0][3]
            pair_strs = [f"{a}\u2194{b}={s}" for a, b, s, _ in score_tuples]
            parts.append(f"Similarity ({method_used}): " + ", ".join(pair_strs))
        output = "\n".join(error_lines + parts)
        return output + "\n" if output else ""

    def process_temporal_grounding(self, query: str, vid_id: str, dic: Dict) -> tuple:
        """时序定位处理"""
        if os.getenv("DISABLE_TEMPORAL_GROUNDING", "0") == "1":
            if not getattr(self, "_tg_disabled_logged", False):
                logger.info("[TG] DISABLE_TEMPORAL_GROUNDING=1; all TG calls will return no-op")
                self._tg_disabled_logged = True
            return ("Temporal grounding is unavailable in this run; use other tools (video_reader, scene_graph, object_tracker, spatial_relation) or answer based on direct reading.", dic)
        duration = dic['duration_map'].get(vid_id, 0)
        
        prompt_version = getattr(self.args, 'prompt_version', 'default')
        if prompt_version == 'minimal':
            agent_prompt = f"Video {vid_id} ({duration}s). Find segments for: {query}"
        else:
            agent_prompt = initial_input_template_temporal_grounding_agent_multivideo_wo_subtitle.format(
                vid_id=vid_id, question=query, duration=duration
            ) if not self.args.use_subtitle else initial_input_template_temporal_grounding_agent_multivideo.format(
                vid_id=vid_id, question=query, duration=duration
            )
        
        messages = [{"role": "user", "content": agent_prompt.replace('thinking>', 'think>')}]
        result = self.single_text2text(messages, self.temporal_grounding_model_name, 
                                       self.temporal_grounding_api_base, self.temporal_grounding_api_keys)
        
        # 解析结果
        answers = re.findall(r"<answer>(.*?)</answer>", result, flags=re.S | re.I)
        intervals = []
        for ans in answers:
            try:
                b_idx, e_idx = ans.find("["), ans.rfind("]")
                if b_idx != -1 and e_idx > b_idx:
                    intervals.extend(robust_eval(ans[b_idx:e_idx+1]))
            except:
                pass
        
        if 'search_history' not in dic:
            dic['search_history'] = []
        dic['search_history'].append({"vid_id": vid_id, "query": query})
        
        return f"Video '{vid_id}' segments for '{query}': {intervals}", dic

    def process_video_segment_retriever_text(self, vid_id: str, query: str, dic: Dict) -> str:
        """文本检索处理"""
        if vid_id not in dic.get("video_map", {}):
            return ""
        
        try:
            path = dic["video_map"][vid_id]
            duration = dic["duration_map"][vid_id]
            topk = int(os.getenv("TOPK", "10"))
            
            clips = self.retriever.get_informative_clips(query, video_path=path, top_k=topk, total_duration=duration)
            indices = [int(v[0].split("/")[-1].split("_")[1]) for v in clips]
            
            return f'<video_segment_retriever id="{vid_id}">{query}</video_segment_retriever> result: {indices}\n'
        except Exception as e:
            logger.error(f"TextRetriever error for {vid_id}: {e}")
            return ""

    def process_subtitle_retriever(self, output_text: str, dic: Dict) -> str:
        """字幕检索处理"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_SUBTITLE_RETRIEVER.findall(output_text)
        if not matches:
            return ""

        results = []
        for vid_id, query in matches:
            if vid_id not in dic['video_map']:
                continue
            
            video_path = dic['video_map'][vid_id]
            duration = dic['duration_map'][vid_id]
            
            try:
                subtitles = self.retriever.get_informative_subtitles(query, video_path=video_path, top_k=10, total_duration=duration)
                formatted = [{'begin': int(x[0]), 'end': int(x[1]), 'text': x[2]} for x in subtitles]
                results.append(f'<subtitle_retriever id="{vid_id}">{query}</subtitle_retriever> result: {formatted}\n')
            except Exception as e:
                logger.error(f"SubtitleRetriever error for {vid_id}: {e}")
        
        return "".join(results)

    def process_subtitle_extractor(self, output_text: str, dic: Dict) -> str:
        """字幕提取处理"""
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[-1]

        matches = RE_SUBTITLE_EXTRACTOR.findall(output_text)
        if not matches:
            return ""

        results = []
        for vid_id, time_range in matches:
            if vid_id not in dic['video_map']:
                continue
            
            video_path = dic['video_map'][vid_id]
            
            for match in time_range.split(';'):
                try:
                    begin, end = float(match.split(':')[0]), float(match.split(':')[1])
                    if 'starting_timestamp_for_subtitles' in dic:
                        begin += dic['starting_timestamp_for_subtitles']
                        end += dic['starting_timestamp_for_subtitles']
                    
                    all_subs = extract_subtitles(video_path)
                    filtered = [{'begin': int(x[0]), 'end': int(x[1]), 'text': x[2]} 
                               for x in all_subs if begin <= x[0] <= end]
                    results.append(f'<subtitle_extractor id="{vid_id}">{match}</subtitle_extractor> result: {filtered}\n')
                except Exception as e:
                    logger.error(f"SubtitleExtractor error: {e}")
        
        return "".join(results)

    # VLM 单次输出的最大字符数（防止 "1 car, 1 car, ..." 幻觉失控污染 context）
    _VLM_OUTPUT_MAX_CHARS = 3000

    def remove_duplicate_sentences(self, text: str) -> str:
        """去除重复句子和逗号片段，并截断过长输出。"""
        if not text:
            return text

        # ── 1) 先按句号/换行去重 ──
        sentences = re.split(r'[.。\n]+', text)
        seen: set = set()
        unique: list = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                unique.append(s)
        text = '. '.join(unique)

        # ── 2) 逗号片段去重（针对 "1 car, 1 car, 1 car, ..." 幻觉） ──
        #    先压连续重复，再压非连续交替重复
        parts = [p.strip() for p in text.split(',')]
        if len(parts) > 3:
            # 2.1 连续重复压缩
            deduped: list = [parts[0]]
            run_count = 1
            for i in range(1, len(parts)):
                if parts[i] == parts[i - 1]:
                    run_count += 1
                    if run_count <= 2:
                        deduped.append(parts[i])
                else:
                    run_count = 1
                    deduped.append(parts[i])

            # 2.2 非连续交替重复压缩（如 car, person, car, person...）
            if len(deduped) > 10:
                seen_counts: dict = {}
                compact: list = []
                for p in deduped:
                    seen_counts[p] = seen_counts.get(p, 0) + 1
                    # 每个片段最多保留 2 次，后续重复直接丢弃
                    if seen_counts[p] <= 2:
                        compact.append(p)
                deduped = compact

            text = ', '.join(deduped)

        # ── 3) 硬截断：防止幻觉输出爆掉 context ──
        if len(text) > self._VLM_OUTPUT_MAX_CHARS:
            text = text[:self._VLM_OUTPUT_MAX_CHARS] + "... [truncated]"
            logger.warning(f"[VLM_TRUNCATE] Output exceeded {self._VLM_OUTPUT_MAX_CHARS} chars, truncated")

        return text


# =============================================================================
# 数据预处理
# =============================================================================

def _letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(max(0, n))]


def _is_mvu_like(data: Any) -> bool:
    if isinstance(data, list) and data:
        sample = data[0]
        return isinstance(sample, dict) and "video_paths" in sample and "question" in sample
    if isinstance(data, dict) and data:
        sample = next(iter(data.values()))
        return isinstance(sample, dict) and "video_paths" in sample and "question" in sample
    return False


def _normalize_mvu_data(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        def _key_sort(x):
            try:
                return int(x)
            except Exception:
                return str(x)
        items = [data[k] for k in sorted(data.keys(), key=_key_sort)]
    else:
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "answer": item.get("ground_truth", item.get("answer", "")),
            "video_paths": item.get("video_paths", []),
            "task": item.get("task", ""),
        })
    return normalized


def preprocess_mvu_data(raw_dic: Dict[str, Any], args) -> Dict[str, Any]:
    """将 MVU 格式样本转为内部统一结构"""
    video_paths = raw_dic.get("video_paths", []) or []
    options = raw_dic.get("options", []) or []
    question = raw_dic.get("question", "")
    answer = raw_dic.get("answer", raw_dic.get("ground_truth", ""))
    task = raw_dic.get("task", "")

    video_ids = _letters(len(video_paths))
    answer_choices = _letters(len(options)) if options else _letters(len(video_paths))

    video_root = args.video_root or os.path.join(args.dataset_folder, args.dataset)
    video_map = {}
    duration_map = {}
    for vid_id, rel_path in zip(video_ids, video_paths):
        if isinstance(rel_path, dict):
            rel_path = rel_path.get("path") or rel_path.get("file_path") or ""
        rel_path = str(rel_path) if rel_path is not None else ""
        if os.path.isabs(rel_path):
            abs_path = rel_path
        else:
            abs_path = os.path.join(video_root, rel_path)
        video_map[vid_id] = abs_path
        try:
            duration_map[vid_id] = float(_get_video_duration(abs_path))
        except Exception:
            duration_map[vid_id] = float(DEFAULT_VIDEO_DURATION)

    first_video = video_map[video_ids[0]] if video_ids else ""
    return {
        "question": question,
        "options": options,
        "answer": str(answer).strip().upper(),
        "task": task,
        "video_map": video_map,
        "duration_map": duration_map,
        "video_path": first_video,
        "duration": duration_map.get(video_ids[0], DEFAULT_VIDEO_DURATION) if video_ids else DEFAULT_VIDEO_DURATION,
        "video_ids": video_ids,
        "answer_choices": answer_choices,
        "no_ref": True,
        "raw_data": raw_dic,
    }

def preprocess_multivideo_data(raw_dic):
    """预处理多视频数据"""
    video_map = {}
    duration_map = {}
    video_text_map = {}

    # Reference
    ref_meta = raw_dic["metadata"]["reference_gt"]
    video_map["ref"] = ref_meta["file_path"]
    # Always read the real video duration inside agent.
    try:
        duration_map["ref"] = float(_get_video_duration(video_map["ref"]))
    except Exception:
        duration_map["ref"] = float(DEFAULT_VIDEO_DURATION)
    
    ref_text = {k: ref_meta[k] for k in ["cf_question", "cf_option1", "cf_option2"] if k in ref_meta}
    if ref_text:
        video_text_map["ref"] = ref_text

    # Options A/B/C/D
    options_meta = raw_dic["metadata"]["options_gt"]
    def _label_sort_key(x: str) -> tuple:
        xu = str(x).strip().upper()
        if xu in ANSWER_LIST:
            return (0, ANSWER_LIST.index(xu))
        return (1, xu)

    for label in sorted(list(options_meta.keys()), key=_label_sort_key):
        opt = options_meta[label]
        video_map[label] = opt["file_path"]
        try:
            duration_map[label] = float(_get_video_duration(video_map[label]))
        except Exception:
            duration_map[label] = float(DEFAULT_VIDEO_DURATION)
        
        opt_text = {k: opt[k] for k in ["cf_question", "cf_option1", "cf_option2"] if k in opt}
        if opt_text:
            video_text_map[label] = opt_text

    def _infer_answer_choices(answer_str: str, option_keys: List[str]) -> List[str]:
        a = str(answer_str or "").strip()
        au = a.upper()
        al = a.lower()
        if al in {"yes", "no"}:
            return ["Yes", "No"]
        if au in {"STRENGTHENER", "WEAKENER"}:
            return ["Strengthener", "Weakener"]
        if au in {"OPTION 1", "OPTION 2"}:
            return ["Option 1", "Option 2"]
        # Default: if options are labeled, treat as MCQ labels.
        keys = [str(k).strip().upper() for k in option_keys if str(k).strip()]
        if keys:
            return keys
        return list(ANSWER_LIST)

    ans0 = raw_dic.get("answer", "")
    if isinstance(ans0, list):
        ans0 = ans0[0] if ans0 else ""
    ans0 = str(ans0).strip()

    # Normalize answer to canonical form.
    if ans0.upper() in ANSWER_LIST:
        ans = ans0.upper()
    elif ans0.lower() in {"yes", "no"}:
        ans = "Yes" if ans0.lower() == "yes" else "No"
    elif ans0.upper() in {"STRENGTHENER", "WEAKENER"}:
        ans = "Strengthener" if ans0.upper() == "STRENGTHENER" else "Weakener"
    else:
        m = re.fullmatch(r"option\\s*([12])", ans0, flags=re.IGNORECASE)
        if m:
            ans = f"Option {m.group(1)}"
        else:
            ans = ans0

    # Explicitly set variable video_ids/answer_choices (avoid assuming A/B/C/D for pair tasks).
    video_ids = ["ref"] if "ref" in video_map else []
    for k in sorted([x for x in video_map.keys() if x != "ref"], key=lambda x: (x not in ANSWER_LIST, x)):
        video_ids.append(k)
    answer_choices = _infer_answer_choices(ans, list(options_meta.keys()))

    return {
        "question": raw_dic.get("question", ""),
        "answer": ans,
        "task": raw_dic.get("task", ""),
        "video_map": video_map,
        "duration_map": duration_map,
        "video_text_map": video_text_map,
        "video_path": video_map["ref"],
        "duration": duration_map["ref"],
        "video_ids": video_ids,
        "answer_choices": answer_choices,
        "raw_data": raw_dic,
    }


def process_dic(dic, args):
    """验证数据有效性"""
    try:
        if isinstance(dic, dict) and "metadata" in dic:
            dic = preprocess_multivideo_data(dic)
        elif isinstance(dic, dict) and "video_paths" in dic:
            dic = preprocess_mvu_data(dic, args)
        else:
            print("process_dic: unknown format, skip")
            return None
        folder_path = f'{args.dataset_folder}/{args.dataset}/embeddings/{args.clip_duration}/large/'
        dic['searchable_map'] = {}

        for vid_id, path in dic['video_map'].items():
            if not os.path.exists(path):
                print(f"Video missing: {path}")
                return None
            
            video_name = os.path.splitext(os.path.basename(path))[0]
            dic['searchable_map'][vid_id] = os.path.exists(f'{folder_path}/{video_name}.pkl')

        return dic
    except Exception as e:
        print(f"process_dic error: {e}")
        return None


def check_valid_data(data_li, args):
    """检查数据有效性"""
    valid = []
    for dic in tqdm(data_li, desc='Checking data'):
        result = process_dic(dic, args)
        if result:
            valid.append(result)
    print(f"Valid: {len(valid)}/{len(data_li)}")
    return valid


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mlvu')
    parser.add_argument('--dataset_mode', type=str, default='')
    parser.add_argument('--dataset_folder', type=str, default='')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--video_root', type=str, default='')
    # NOTE: This is NOT the video length. It's the segmentation length (seconds) used by
    # retriever/embeddings cache paths, e.g. .../embeddings/{clip_duration}/...
    parser.add_argument(
        '--clip_duration',
        type=int,
        default=10,
        help='Segmentation length (seconds) for retriever/embeddings cache. NOT video duration.',
    )
    parser.add_argument(
        '--segment_duration',
        dest='clip_duration',
        type=int,
        help='Alias of --clip_duration (segmentation length, not video duration).',
    )
    parser.add_argument('--topk_per_query', type=int, default=1)
    parser.add_argument('--retriever_type', type=str, default='large')
    parser.add_argument('--thread_idx', type=int, default=0)
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--begin_sample_number', type=int, default=0)
    parser.add_argument('--end_sample_number', type=int, default=10000000000)
    parser.add_argument('--random_shuffle', action='store_true')
    parser.add_argument('--overwrite_output', type=int, default=0)
    parser.add_argument('--use_subtitle', type=int, default=0)
    parser.add_argument('--clip_fps', type=float, default=2)
    parser.add_argument('--coarse_fps', type=float, default=1.0)
    parser.add_argument('--fine_fps', type=float, default=0.0)
    parser.add_argument('--enable_scene_detector', type=int, default=1)
    parser.add_argument('--enable_object_tracker', type=int, default=1)
    parser.add_argument('--enable_spatial_relation', type=int, default=1)
    parser.add_argument('--enable_scene_graph', type=int, default=1)
    parser.add_argument('--enable_visual_similarity', type=int, default=1)
    parser.add_argument('--visual_similarity_model', type=str, default='clip', choices=['color_hist', 'hist', 'clip', 'dinov2'])
    parser.add_argument('--visual_similarity_clip_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--visual_similarity_dinov2_model', type=str, default='facebook/dinov2-base')
    parser.add_argument('--visual_similarity_device', type=str, default='cuda')
    parser.add_argument('--visual_similarity_max_frames', type=int, default=16)
    parser.add_argument('--scene_graph_max_objects', type=int, default=20)
    parser.add_argument('--scene_graph_max_frames', type=int, default=16)
    parser.add_argument('--max_frames', type=int, default=32,
                        help='Global max frames for video sampling (timestamp_to_clip_path).')
    parser.add_argument('--counting_scene_graph_max_frames', type=int, default=12,
                        help='Max frames for scene_graph in Counting tasks.')
    parser.add_argument('--counting_max_seconds', type=float, default=0.0,
                        help='Max seconds per video_reader range for Counting tasks. 0 disables capping.')
    parser.add_argument('--counting_max_segments', type=int, default=1,
                        help='Max number of video_reader segments per video for Counting.')
    parser.add_argument('--counting_video_reader_fps', type=float, default=0.3,
                        help='FPS for Counting video_reader to reduce frames.')
    parser.add_argument('--counting_video_reader_max_frames', type=int, default=8,
                        help='Max frames for Counting video_reader after sampling.')
    parser.add_argument('--scene_graph_model', type=str, default='groundingdino', choices=['groundingdino', 'yolo'])
    parser.add_argument('--scene_graph_device', type=str, default='cuda')
    parser.add_argument('--groundingdino_model', type=str, default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--groundingdino_box_threshold', type=float, default=0.3)
    parser.add_argument('--groundingdino_text_threshold', type=float, default=0.25)
    parser.add_argument('--auto_select_gpu', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--enable_skills', type=int, default=1)
    parser.add_argument('--skills_dirs', type=str, default='',
                        help='Extra skills dirs, separated by os.pathsep')
    parser.add_argument('--video_reader_caption', type=int, default=1,
                        help='If 1, force video_reader VLM to output CAPTION+EVIDENCE (facts only).')
    parser.add_argument('--pretool_sweep', type=int, default=0,
                        help='If 1, run a lightweight sweep of tools before LLM planning (can be slow).')
    parser.add_argument('--pretool_max_seconds', type=float, default=10.0,
                        help='Max seconds per video for pretool_sweep ranges.')
    parser.add_argument('--pretool_fps', type=float, default=2.0,
                        help='FPS used for pretool_sweep vision tools (enables frame cache reuse).')
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--skip_data_check', type=int, default=0,
                        help='If 1, skip upfront dataset validation for faster startup.')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--prompt_version', type=str, default='default', choices=['default', 'short', 'minimal'])
    parser.add_argument('--no_ref_prompt_version', type=str, default='current',
                        choices=['current', 'legacy'],
                        help='Prompt version for no_ref tasks (legacy matches best run).')
    parser.add_argument('--direct_vlm_threshold', type=int, default=0,
                        help='Gate for baseline-style direct VLM. '
                             '0=off (default). N>0: n_videos<=N → direct-first (return immediately); '
                             'n_videos>N → agent-first with direct hint + fallback.')
    args = parser.parse_args()

    if args.visual_similarity_device == "cuda" and not torch.cuda.is_available():
        args.visual_similarity_device = "cpu"
    if args.scene_graph_device == "cuda" and not torch.cuda.is_available():
        args.scene_graph_device = "cpu"

    # Apply global max_frames override for all frame samplers.
    try:
        from . import video_utils_base, video_utils_frames, vision_tools, video_utils
        mf = int(args.max_frames)
        if mf > 0:
            video_utils_base.MAX_FRAMES = mf
            video_utils_frames.MAX_FRAMES = mf
            vision_tools.MAX_FRAMES = mf
            video_utils.MAX_FRAMES = mf
    except Exception as e:
        logger.warning(f"Failed to set MAX_FRAMES override: {e}")

    # 配置保存路径
    vlm = os.getenv('API_MODEL_NAME_VLM', 'unknown').split('/')[-1]
    lrm = '_'.join(os.getenv('API_MODEL_NAME', 'unknown').split('/')[-2:])
    temp = '_'.join(os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING', 'unknown').split('/')[-2:])
    topk = os.getenv('TOPK', '10')

    if not args.save_path:
        args.save_path = f"./eval_result/{args.dataset}{args.dataset_mode}_LRM_{lrm}_VLM_{vlm}_TOPK_{topk}.json.part{args.thread_idx}"
    else:
        args.save_path = f"{args.save_path}.part{args.thread_idx}"
    
    os.makedirs('./eval_result', exist_ok=True)

    # 数据集特定配置
    if args.dataset == 'LongVideoBench':
        args.use_subtitle = 1
    elif args.dataset in ['mlvu', 'lvbench']:
        args.use_subtitle = 0

    print(f"Save path: {args.save_path}")

    # 加载数据
    read_path = args.input_path
    if not read_path:
        default_path = f'{args.dataset_folder}/{args.dataset}/qa{args.dataset_mode}.json'
        mvu_path = f'{args.dataset_folder}/{args.dataset}/MVU_Eval_QAs.json'
        if args.dataset.lower() in {"mvu", "mvueval", "mvu_eval"} and os.path.exists(mvu_path):
            read_path = mvu_path
        else:
            read_path = default_path

    data_raw = json.load(open(read_path))
    if _is_mvu_like(data_raw):
        data_li = _normalize_mvu_data(data_raw)
    else:
        data_li = data_raw
    
    if args.tasks != 'all':
        data_li = [d for d in data_li if d.get('task') in args.tasks.split(',')]
    
    data_li = data_li[args.begin_sample_number:args.end_sample_number]
    if int(getattr(args, "skip_data_check", 0)) == 1:
        print("Skip data check: enabled")
        # 只做最基本预处理，跳过昂贵的逐条文件/embedding存在性检查
        processed = []
        for dic in data_li:
            try:
                if isinstance(dic, dict) and "metadata" in dic:
                    out = preprocess_multivideo_data(dic)
                elif isinstance(dic, dict) and "video_paths" in dic:
                    out = preprocess_mvu_data(dic, args)
                else:
                    continue
                # 与 process_dic 对齐：补 searchable_map 字段
                out['searchable_map'] = {vid_id: False for vid_id in out.get('video_map', {}).keys()}
                processed.append(out)
            except Exception:
                continue
        data_li = processed
    else:
        data_li = check_valid_data(data_li, args)

    # 分片处理
    chunk_size = len(data_li) // args.thread_num + 1
    data_li = data_li[args.thread_idx * chunk_size:(args.thread_idx + 1) * chunk_size]

    print(f"Processing {len(data_li)} samples")

    # 加载已处理结果
    processed_results = []
    processed_questions = set()
    jsonl_path = _jsonl_path_for_save(args.save_path)
    
    if os.path.exists(args.save_path) and not args.overwrite_output:
        try:
            for file in glob.glob(args.save_path.split('.part')[0] + '.part*'):
                with open(file, 'r') as f:
                    results = json.load(f).get('processed_results', [])
                    for r in results:
                        if r.get('score') is not None and r not in processed_results:
                            processed_results.append(r)
            
            for r in processed_results:
                rd = r.get("raw_data", {})
                key = rd.get("question", "") + "\x00" + rd.get("video_path", rd.get("video_map", {}).get("ref", ""))
                processed_questions.add(key)

            print(f"Loaded {len(processed_results)} previous results (json)")
        except Exception as e:
            print(f"Load error: {e}")
    elif os.path.exists(jsonl_path) and not args.overwrite_output:
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get('score') is not None and r not in processed_results:
                        processed_results.append(r)
            for r in processed_results:
                rd = r.get("raw_data", {})
                key = rd.get("question", "") + "\x00" + rd.get("video_path", rd.get("video_map", {}).get("ref", ""))
                processed_questions.add(key)
            print(f"Loaded {len(processed_results)} previous results (jsonl)")
        except Exception as e:
            print(f"Load jsonl error: {e}")

    # 过滤已处理
    unprocessed = []
    for dic in data_li:
        key = dic.get("question", "") + "\x00" + dic.get("video_path", dic.get("video_map", {}).get("ref", ""))
        if key not in processed_questions:
            unprocessed.append(dic)

    print(f"To process: {len(unprocessed)}")

    if not unprocessed:
        print("All done!")
        exit(0)

    if args.random_shuffle:
        random.shuffle(unprocessed)

    # 初始化处理器
    manager = SingleSampleProcessor(args)

    print("Mode: STANDARD")

    # 主循环
    checkpoint_interval = max(1, int(getattr(args, "checkpoint_interval", 5)))
    start_time = time.time()
    
    for idx, dic in enumerate(tqdm(unprocessed, desc="Processing")):
        try:
            result = manager.process_single_sample(dic)
        except Exception as e:
            # 和 baseline 一致：报错 → log → continue（不记为 score=False）
            logger.error(f"Sample {idx} error: {e}")
            logger.error(traceback.format_exc())
            continue

        # 和 baseline 对齐：只有产出 pred_answer 的 case 才写入文件。
        # 没有答案的不写入 → eval 分母 = 文件行数 = 成功推理数，口径一致。
        has_answer = bool(result and result.get('raw_data', {}).get('pred_answer'))
        if result and has_answer:
            processed_results.append(result)
            try:
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Append jsonl error: {e}")
        elif result:
            logger.warning(f"[SKIP_WRITE] Sample {idx}: no pred_answer, skipped")

        # 定期保存
        if (idx + 1) % checkpoint_interval == 0 or idx == len(unprocessed) - 1:
            total = len(processed_results)
            correct = sum(1 for r in processed_results if r.get('score'))
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (len(unprocessed) - idx - 1) / 60
            
            logger.info(f"[{idx+1}/{len(unprocessed)}] Acc: {correct}/{total} ({correct/max(total,1)*100:.1f}%) ETA: {eta:.1f}min")

    # 最终统计
    print(f"\n{'='*50}")
    print(f"Completed: {len(processed_results)} samples")
    if processed_results:
        correct = sum(1 for r in processed_results if r.get('score'))
        print(f"Accuracy: {correct/len(processed_results)*100:.2f}% ({correct}/{len(processed_results)})")
        
        # 按任务统计
        task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in processed_results:
            task = r.get('raw_data', {}).get('task', 'unknown')
            if isinstance(task, list):
                task = task[0]
            task_stats[task]['total'] += 1
            if r.get('score'):
                task_stats[task]['correct'] += 1
        
        if len(task_stats) > 1:
            print("\nPer-task:")
            for task, stats in sorted(task_stats.items()):
                acc = stats['correct'] / stats['total'] * 100 if stats['total'] else 0
                print(f"  {task}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
