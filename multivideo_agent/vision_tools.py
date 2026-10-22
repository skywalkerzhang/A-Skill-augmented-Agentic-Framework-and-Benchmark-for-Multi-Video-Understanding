import math
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .video_utils_base import MAX_FRAMES


# =============================================================================
# 颜色 / 属性检测（传统CV，无幻觉）
# =============================================================================

# HSV 色相映射表 (H in OpenCV: 0-179)
_COLOR_RANGES: List[Tuple[str, Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = [
    # name, (H_lo, H_hi), (S_lo, S_hi), (V_lo, V_hi)
    ("red",      (0, 10),    (70, 255),  (50, 255)),
    ("red",      (170, 179), (70, 255),  (50, 255)),
    ("orange",   (11, 25),   (70, 255),  (50, 255)),
    ("yellow",   (26, 34),   (70, 255),  (50, 255)),
    ("green",    (35, 85),   (40, 255),  (40, 255)),
    ("cyan",     (86, 100),  (70, 255),  (50, 255)),
    ("blue",     (101, 130), (70, 255),  (50, 255)),
    ("purple",   (131, 160), (40, 255),  (40, 255)),
    ("pink",     (161, 169), (30, 255),  (50, 255)),
]

_ACHROMATIC_S_THRESH = 40   # S < 此值 → 无彩色
_ACHROMATIC_V_BLACK  = 50   # V < 此值 → 黑色
_ACHROMATIC_V_WHITE  = 200  # V > 此值 → 白色


def _dominant_color_name(crop_bgr: np.ndarray) -> str:
    """从裁剪的 BGR 图像分析主色调，返回颜色名称字符串。
    纯传统 CV，不依赖任何模型。"""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 先统计无彩色像素
    achromatic = s < _ACHROMATIC_S_THRESH
    n_total = max(h.size, 1)
    n_achromatic = int(achromatic.sum())

    if n_achromatic > n_total * 0.6:
        # 大部分无彩色
        mean_v = float(v[achromatic].mean()) if n_achromatic else float(v.mean())
        if mean_v < _ACHROMATIC_V_BLACK:
            return "black"
        if mean_v > _ACHROMATIC_V_WHITE:
            return "white"
        return "gray"

    # 有彩色像素：按色相范围投票
    chromatic_mask = ~achromatic
    ch = h[chromatic_mask]
    cs = s[chromatic_mask]
    cv_ = v[chromatic_mask]
    votes: Counter = Counter()
    for name, (h_lo, h_hi), (s_lo, s_hi), (v_lo, v_hi) in _COLOR_RANGES:
        mask = (ch >= h_lo) & (ch <= h_hi) & (cs >= s_lo) & (cs <= s_hi) & (cv_ >= v_lo) & (cv_ <= v_hi)
        votes[name] += int(mask.sum())
    if not votes:
        return ""
    return votes.most_common(1)[0][0]


def _detect_color_for_box(image_bgr: np.ndarray, box: List[float]) -> str:
    """给定一张 BGR 图和 xyxy 坐标，裁剪出 bbox 区域并返回主色调名称。"""
    h_img, w_img = image_bgr.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w_img, int(box[2]))
    y2 = min(h_img, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = image_bgr[y1:y2, x1:x2]
    return _dominant_color_name(crop)


# 哪些目标类别需要做颜色检测（优先级高 = 对推理价值大）
_COLOR_DETECT_LABELS = {
    "traffic light", "traffic signal", "light",
    "car", "truck", "bus", "motorcycle", "bicycle",
    "shirt", "dress", "hat", "jacket", "bag",
    "ball", "balloon", "flag", "sign",
}


def _should_detect_color(label: str) -> bool:
    """判断该目标是否值得做颜色检测。"""
    ll = label.lower().strip()
    return any(kw in ll for kw in _COLOR_DETECT_LABELS)


# =============================================================================
# 坐标 → 自然语言位置描述
# =============================================================================

def _box_to_position(box: List[float], img_w: float, img_h: float) -> str:
    """将 xyxy 坐标转为 'upper-left' / 'center' / 'bottom-right' 等描述。"""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    col = "left" if cx < img_w / 3 else ("right" if cx > img_w * 2 / 3 else "center")
    row = "upper" if cy < img_h / 3 else ("lower" if cy > img_h * 2 / 3 else "middle")
    if col == "center" and row == "middle":
        return "center"
    if col == "center":
        return row
    if row == "middle":
        return col
    return f"{row}-{col}"


def _box_to_size(area: float, img_area: float) -> str:
    """将面积占比转为 small / medium / large。"""
    if img_area <= 0:
        return ""
    ratio = area / img_area
    if ratio > 0.10:
        return "large"
    if ratio > 0.02:
        return "medium"
    return "small"


# =============================================================================
# 场景图 → 自然语言摘要
# =============================================================================

def _summarize_scene_graph_frame(
    nodes: List[Dict],
    relations: List[Dict],
    timestamp: float,
    img_w: float = 1920.0,
    img_h: float = 1080.0,
) -> str:
    """将单帧的节点+关系列表压缩成自然语言摘要。"""
    if not nodes:
        return f"t={timestamp:.1f}s: no objects detected"

    img_area = img_w * img_h

    # ---- 按 label 分组 ----
    from collections import defaultdict as _dd
    label_groups: Dict[str, List[Dict]] = _dd(list)
    for n in nodes:
        label_groups[n["label"]].append(n)

    parts: List[str] = []
    for label, group in sorted(label_groups.items(), key=lambda x: -len(x[1])):
        count = len(group)
        # 收集颜色
        colors = [n.get("color", "") for n in group if n.get("color")]
        color_counter = Counter(colors)
        # 收集尺寸
        sizes = [_box_to_size(n.get("area", 0), img_area) for n in group]
        size_counter = Counter(s for s in sizes if s)
        # 收集位置
        positions = [n.get("position", "") for n in group if n.get("position")]
        pos_counter = Counter(positions)

        desc_parts: List[str] = []
        if count == 1:
            n = group[0]
            attrs = []
            if n.get("color"):
                attrs.append(n["color"])
            sz = _box_to_size(n.get("area", 0), img_area)
            if sz:
                attrs.append(sz)
            pos = n.get("position", "")
            attr_str = " ".join(attrs)
            item = f"{attr_str} {label}".strip() if attr_str else label
            if pos:
                item += f" ({pos})"
            desc_parts.append(item)
        else:
            # 多个同类目标：聚合描述
            if color_counter:
                for color, cnt in color_counter.most_common(3):
                    if cnt == count:
                        desc_parts.append(f"{count} {label} ({color})")
                    else:
                        desc_parts.append(f"{cnt} {color}")
                if not desc_parts:
                    desc_parts.append(f"{count} {label}")
            else:
                desc_parts.append(f"{count} {label}")
            # 尺寸/位置只在数量少时附加
            if count <= 3 and size_counter:
                sz_info = ", ".join(f"{c} {s}" for s, c in size_counter.most_common())
                desc_parts.append(f"[{sz_info}]")

        parts.append(", ".join(desc_parts))

    # ---- 只保留跨类别的有意义关系 ----
    label_map = {n.get("id", i): n["label"] for i, n in enumerate(nodes)}
    meaningful_rels: List[str] = []
    for r in relations:
        subj_label = label_map.get(r.get("subj"), "?")
        obj_label = label_map.get(r.get("obj"), "?")
        pred = r.get("pred", "")
        # 跳过同类关系 (traffic light right_of traffic light 没意义)
        if subj_label == obj_label:
            continue
        # 跳过 left_of / right_of 这种低信息量的位置关系
        if pred in ("left_of", "right_of", "above", "below"):
            continue
        # 保留 near / overlaps 等有意义的关系
        subj_color = ""
        obj_color = ""
        for n in nodes:
            if n.get("id") == r.get("subj") and n.get("color"):
                subj_color = n["color"] + " "
            if n.get("id") == r.get("obj") and n.get("color"):
                obj_color = n["color"] + " "
        meaningful_rels.append(f"{subj_color}{subj_label} {pred} {obj_color}{obj_label}")

    summary = f"t={timestamp:.1f}s: " + "; ".join(parts)
    if meaningful_rels:
        summary += " | relations: " + "; ".join(meaningful_rels[:5])
    return summary


def summarize_scene_graph_results(
    raw_results: List[Dict],
    img_w: float = 1920.0,
    img_h: float = 1080.0,
) -> str:
    """将多帧场景图结果列表转为紧凑的自然语言字符串。"""
    if not raw_results:
        return "no detections"
    if len(raw_results) == 1 and "error" in raw_results[0]:
        return str(raw_results[0])

    lines: List[str] = []
    for frame in raw_results:
        if "error" in frame:
            lines.append(str(frame))
            continue
        ts = frame.get("time", 0.0)
        nodes = frame.get("nodes", [])
        relations = frame.get("relations", [])
        lines.append(_summarize_scene_graph_frame(nodes, relations, ts, img_w, img_h))
    return "\n".join(lines)


def _parse_range(range_text: Optional[str], default_begin: float, default_end: float) -> Tuple[float, float]:
    if not range_text:
        return default_begin, default_end
    if ":" not in range_text:
        return default_begin, default_end
    try:
        begin_s, end_s = range_text.split(":", 1)
        return float(begin_s), float(end_s)
    except ValueError:
        return default_begin, default_end


def _iter_sampled_frames(video_path: str, sample_fps: float) -> Tuple[float, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if video_fps <= 0:
        video_fps = 25.0
    step = max(int(round(video_fps / max(sample_fps, 0.1))), 1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / video_fps
            yield timestamp, frame
        frame_idx += 1
    cap.release()


def detect_scene_changes(
    video_path: str,
    sample_fps: float = 1.0,
    hist_bins: int = 32,
    threshold: float = 0.6,
    min_gap: float = 0.5,
) -> List[float]:
    changes: List[float] = []
    last_change_t = -math.inf
    prev_hist = None

    for timestamp, frame in _iter_sampled_frames(video_path, sample_fps):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [hist_bins] * 3, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            distance = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance >= threshold and (timestamp - last_change_t) >= min_gap:
                changes.append(round(float(timestamp), 2))
                last_change_t = timestamp
        prev_hist = hist

    return changes


def _hist_feature(image_path: str, bins: int = 32) -> Optional[np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins] * 3, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _mean_feature(frame_paths: List[str], bins: int = 32) -> Optional[np.ndarray]:
    feats = []
    for p in frame_paths:
        feat = _hist_feature(p, bins=bins)
        if feat is not None:
            feats.append(feat)
    if not feats:
        return None
    return np.mean(np.stack(feats, axis=0), axis=0)


def visual_similarity_from_frames(
    frames_a: List[str],
    frames_b: List[str],
    bins: int = 32,
) -> Tuple[float, str]:
    feat_a = _mean_feature(frames_a, bins=bins)
    feat_b = _mean_feature(frames_b, bins=bins)
    if feat_a is None or feat_b is None:
        return 0.0, "no_valid_frames"
    denom = float(np.linalg.norm(feat_a) * np.linalg.norm(feat_b))
    if denom <= 0:
        return 0.0, "degenerate_feature"
    score = float(np.dot(feat_a, feat_b) / denom)
    return max(min(score, 1.0), -1.0), "color_hist"


@lru_cache(maxsize=2)
def _load_yolo(model_name: str):
    from ultralytics import YOLO

    return YOLO(model_name)


def run_object_detection(
    frame_paths: List[str],
    timestamps: List[float],
    target: Optional[str] = None,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    max_frames: int = MAX_FRAMES,
) -> List[Dict]:
    results: List[Dict] = []
    if not frame_paths:
        return results

    if len(frame_paths) > max_frames:
        step = max(len(frame_paths) // max_frames, 1)
        frame_paths = frame_paths[::step][:max_frames]
        timestamps = timestamps[::step][:max_frames]

    if target:
        targets = [t.strip().lower() for t in str(target).split(",") if t.strip()]
    else:
        targets = None

    try:
        model = _load_yolo(model_name)
    except Exception as exc:
        return [{"error": f"YOLO unavailable: {exc}"}]

    for path, ts in zip(frame_paths, timestamps):
        try:
            preds = model.predict(path, conf=conf, verbose=False)
        except Exception:
            continue
        if not preds:
            continue
        det = preds[0]
        if det.boxes is None:
            continue
        names = det.names or {}
        objects = []
        boxes = []
        for box in det.boxes:
            cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            label = names.get(cls_id, str(cls_id))
            if targets and not any(t in label.lower() for t in targets):
                continue
            xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else []
            objects.append(label)
            boxes.append([round(float(v), 2) for v in xyxy])
        if objects:
            results.append({"time": round(float(ts), 2), "objects": objects, "boxes": boxes})
    return results


def summarize_object_detection(
    raw_results: List[Dict],
    img_w: float = 1920.0,
    img_h: float = 1080.0,
    frame_paths: Optional[List[str]] = None,
    timestamps: Optional[List[float]] = None,
) -> str:
    """将 run_object_detection 的结果转为紧凑自然语言摘要。

    - 聚合重复 label（"6 traffic lights" 而不是列6遍）
    - box 坐标 → 位置描述（upper-left / center 等）
    - 自动检测颜色（裁剪 bbox 区域分析主色调）
    """
    if not raw_results:
        return "no detections"
    if len(raw_results) == 1 and "error" in raw_results[0]:
        return str(raw_results[0])

    img_area = img_w * img_h
    lines: List[str] = []

    # 建立 timestamp → frame_path 映射，用于精确匹配每帧做颜色检测
    _ts_to_path: Dict[float, str] = {}
    if frame_paths and timestamps and len(frame_paths) == len(timestamps):
        for fp, t in zip(frame_paths, timestamps):
            _ts_to_path[round(float(t), 2)] = fp
    _img_cache: Dict[str, Optional[np.ndarray]] = {}

    def _get_frame_img(det_ts: float) -> Optional[np.ndarray]:
        # 精确匹配
        path = _ts_to_path.get(round(det_ts, 2))
        if not path and _ts_to_path:
            # 找最近的帧
            best_t = min(_ts_to_path.keys(), key=lambda t: abs(t - det_ts))
            if abs(best_t - det_ts) < 1.0:
                path = _ts_to_path[best_t]
        if not path and frame_paths:
            path = frame_paths[0]
        if not path:
            return None
        if path not in _img_cache:
            _img_cache[path] = cv2.imread(path)
        return _img_cache[path]

    from collections import defaultdict as _dd

    for det in raw_results:
        if "error" in det:
            lines.append(str(det))
            continue
        ts = det.get("time", 0.0)
        objects = det.get("objects", [])
        boxes = det.get("boxes", [])
        if not objects:
            continue

        # 按需读取对应帧图像
        frame_img = _get_frame_img(ts)
        local_img_w, local_img_h = img_w, img_h
        if frame_img is not None:
            local_img_h, local_img_w = frame_img.shape[:2]
        local_img_area = local_img_w * local_img_h

        # 按 label 分组
        label_groups: Dict[str, List[List[float]]] = _dd(list)
        for label, box in zip(objects, boxes):
            label_groups[label].append(box)

        parts: List[str] = []
        for label, grp_boxes in sorted(label_groups.items(), key=lambda x: -len(x[1])):
            count = len(grp_boxes)
            # 颜色检测
            colors: List[str] = []
            if frame_img is not None and _should_detect_color(label):
                for box in grp_boxes:
                    c = _detect_color_for_box(frame_img, box)
                    if c:
                        colors.append(c)
            color_counter = Counter(colors)
            # 位置
            positions = [_box_to_position(box, local_img_w, local_img_h) for box in grp_boxes]
            pos_counter = Counter(positions)

            if count == 1:
                attrs: List[str] = []
                if color_counter:
                    attrs.append(color_counter.most_common(1)[0][0])
                sz = _box_to_size(
                    max(grp_boxes[0][2] - grp_boxes[0][0], 0) * max(grp_boxes[0][3] - grp_boxes[0][1], 0),
                    local_img_area,
                )
                if sz:
                    attrs.append(sz)
                attr_str = " ".join(attrs)
                item = f"{attr_str} {label}".strip() if attr_str else label
                if positions[0]:
                    item += f" ({positions[0]})"
                parts.append(item)
            else:
                # 聚合描述
                if color_counter:
                    top_color = color_counter.most_common(1)[0]
                    if top_color[1] == count:
                        parts.append(f"{count} {label} ({top_color[0]})")
                    else:
                        color_desc = ", ".join(f"{cnt} {clr}" for clr, cnt in color_counter.most_common(3))
                        parts.append(f"{count} {label} ({color_desc})")
                else:
                    parts.append(f"{count} {label}")
                # 位置分布（简洁）
                if count <= 4:
                    pos_desc = ", ".join(pos_counter.keys())
                    if pos_desc:
                        parts[-1] += f" [{pos_desc}]"
                else:
                    regions = sorted(set(pos_counter.keys()))
                    if regions:
                        parts[-1] += f" [spread across {', '.join(regions)}]"

        lines.append(f"t={ts:.1f}s: " + "; ".join(parts))

    return "\n".join(lines)


def _pairs_to_count(n_pairs: int) -> int:
    """从 C(n,2) = n_pairs 反推 n（取整）。C(n,2) = n*(n-1)/2。"""
    if n_pairs <= 0:
        return 0
    # n = (1 + sqrt(1 + 8*pairs)) / 2
    n = (1.0 + math.sqrt(1.0 + 8.0 * n_pairs)) / 2.0
    return max(2, round(n))


def summarize_spatial_relations(
    raw_results: List[Dict],
) -> str:
    """将 run_spatial_relations 的结果转为紧凑自然语言摘要。

    - 过滤掉同类物体之间的关系（"traffic light left_of traffic light" 无信息量）
    - 只保留跨类别的有意义关系
    - 同类物体只做聚合计数（用组合数反推真实数量）
    """
    if not raw_results:
        return "no spatial relations detected"
    if len(raw_results) == 1 and "error" in raw_results[0]:
        return str(raw_results[0])

    _REL_WORDS = {"left_of", "right_of", "above", "below", "overlaps", "near"}

    lines: List[str] = []
    for det in raw_results:
        if "error" in det:
            lines.append(str(det))
            continue
        ts = det.get("time", 0.0)
        relations = det.get("relations", [])
        if not relations:
            continue

        # 分类：跨类 vs 同类
        cross_class: List[str] = []
        same_class_pairs: Counter = Counter()
        for rel_str in relations:
            tokens = rel_str.split()
            rel_idx = None
            for i, tok in enumerate(tokens):
                if tok in _REL_WORDS:
                    rel_idx = i
                    break
            if rel_idx is None:
                cross_class.append(rel_str)
                continue
            subj = " ".join(tokens[:rel_idx])
            obj = " ".join(tokens[rel_idx + 1:])

            if subj.lower() == obj.lower():
                same_class_pairs[subj] += 1
            else:
                cross_class.append(rel_str)

        parts_out: List[str] = []
        # 同类聚合：用组合数反推真实数量
        for label, n_pairs in same_class_pairs.most_common():
            n_objs = _pairs_to_count(n_pairs)
            parts_out.append(f"{n_objs} {label} detected")
        # 跨类关系（去重，最多 5 条）
        seen: set = set()
        for r in cross_class:
            if r not in seen:
                seen.add(r)
                parts_out.append(r)
            if len(parts_out) >= 8:
                break

        if parts_out:
            lines.append(f"t={ts:.1f}s: " + "; ".join(parts_out))

    return "\n".join(lines) if lines else "no meaningful spatial relations"


def _load_clip(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoProcessor, CLIPModel
    except Exception:
        return None, None
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def _load_dinov2(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception:
        return None, None
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def _load_groundingdino(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except Exception:
        return None, None
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def _image_embeddings(
    frame_paths: List[str],
    model_type: str,
    model_name: str,
    device: str,
    max_frames: int,
) -> Optional[np.ndarray]:
    try:
        import torch
    except Exception:
        return None

    if len(frame_paths) > max_frames:
        step = max(len(frame_paths) // max_frames, 1)
        frame_paths = frame_paths[::step][:max_frames]

    images = []
    for p in frame_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not images:
        return None

    if model_type == "clip":
        model, processor = _load_clip(model_name, device)
        if model is None:
            return None
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
    elif model_type == "dinov2":
        model, processor = _load_dinov2(model_name, device)
        if model is None:
            return None
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            feats = outputs.last_hidden_state.mean(dim=1)
    else:
        return None

    feats = feats / torch.clamp(feats.norm(dim=-1, keepdim=True), min=1e-6)
    return feats.cpu().numpy()


def visual_similarity_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0
    vec_a = emb_a.mean(axis=0)
    vec_b = emb_b.mean(axis=0)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0 else 0.0


def _pair_relation(box_a: List[float], box_b: List[float]) -> str:
    if _bbox_iou(box_a, box_b) >= 0.1:
        return "overlaps"
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    cx_a, cy_a = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    cx_b, cy_b = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    dx, dy = cx_b - cx_a, cy_b - cy_a
    if abs(dx) >= abs(dy):
        return "left_of" if dx > 0 else "right_of"
    return "above" if dy > 0 else "below"


def run_spatial_relations(
    frame_paths: List[str],
    timestamps: List[float],
    targets: Optional[str] = None,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    max_frames: int = MAX_FRAMES,
) -> List[Dict]:
    detections = run_object_detection(
        frame_paths,
        timestamps,
        target=targets,
        model_name=model_name,
        conf=conf,
        max_frames=max_frames,
    )
    results: List[Dict] = []
    for det in detections:
        if "error" in det:
            return [det]
        objects = det.get("objects", [])
        boxes = det.get("boxes", [])
        relations = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                rel = _pair_relation(boxes[i], boxes[j])
                relations.append(f"{objects[i]} {rel} {objects[j]}")
        if relations:
            results.append({"time": det.get("time"), "relations": relations})
    return results


def run_scene_graph(
    frame_paths: List[str],
    timestamps: List[float],
    targets: Optional[str] = None,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    max_frames: int = MAX_FRAMES,
    max_objects: int = 20,
    near_thresh: float = 0.25,
) -> List[Dict]:
    detections = run_object_detection(
        frame_paths,
        timestamps,
        target=targets,
        model_name=model_name,
        conf=conf,
        max_frames=max_frames,
    )

    # 建立 frame_path → 原始图像 的映射，用于颜色检测（按需读取）
    _img_cache: Dict[str, Optional[np.ndarray]] = {}

    def _get_image(path: str) -> Optional[np.ndarray]:
        if path not in _img_cache:
            _img_cache[path] = cv2.imread(path)
        return _img_cache[path]

    # 需要将 frame_paths 与 timestamps 同步采样以匹配 detections
    if len(frame_paths) > max_frames:
        step = max(len(frame_paths) // max_frames, 1)
        frame_paths_sampled = frame_paths[::step][:max_frames]
    else:
        frame_paths_sampled = list(frame_paths)

    results: List[Dict] = []
    det_idx = 0
    for det in detections:
        if "error" in det:
            return [det]
        objects = det.get("objects", [])
        boxes = det.get("boxes", [])
        if not objects or not boxes:
            det_idx += 1
            continue

        if len(objects) > max_objects:
            objects = objects[:max_objects]
            boxes = boxes[:max_objects]

        # 尝试获取对应帧图像用于颜色检测
        frame_img = None
        img_w, img_h = 1920.0, 1080.0
        # 按 timestamp 匹配对应的帧
        target_ts = det.get("time", 0.0)
        for fp, ts in zip(frame_paths_sampled, timestamps if len(timestamps) >= len(frame_paths_sampled) else frame_paths_sampled):
            if isinstance(ts, (int, float)) and abs(float(ts) - target_ts) < 1.0:
                frame_img = _get_image(fp)
                break
        if frame_img is None and det_idx < len(frame_paths_sampled):
            frame_img = _get_image(frame_paths_sampled[det_idx])
        if frame_img is not None:
            img_h, img_w = frame_img.shape[:2]

        nodes = []
        for idx, (label, box) in enumerate(zip(objects, boxes)):
            x1, y1, x2, y2 = box
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            node: Dict = {"id": idx, "label": label, "area": round(area, 2)}
            # 颜色检测
            if frame_img is not None and _should_detect_color(label):
                color = _detect_color_for_box(frame_img, box)
                if color:
                    node["color"] = color
            # 位置描述（替代原始坐标）
            node["position"] = _box_to_position(box, img_w, img_h)
            nodes.append(node)

        relations = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                box_a = boxes[i]
                box_b = boxes[j]
                rel = _pair_relation(box_a, box_b)
                relations.append({"subj": i, "pred": rel, "obj": j})

                ax1, ay1, ax2, ay2 = box_a
                bx1, by1, bx2, by2 = box_b
                cx_a, cy_a = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
                cx_b, cy_b = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                diag_a = math.hypot(ax2 - ax1, ay2 - ay1)
                diag_b = math.hypot(bx2 - bx1, by2 - by1)
                dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
                if dist <= near_thresh * max(diag_a, diag_b, 1.0):
                    relations.append({"subj": i, "pred": "near", "obj": j})

                iou = _bbox_iou(box_a, box_b)
                if iou >= 0.3:
                    relations.append({"subj": i, "pred": "overlaps", "obj": j})

        results.append({"time": det.get("time"), "nodes": nodes, "relations": relations})
        det_idx += 1
    return results


def run_scene_graph_grounded(
    frame_paths: List[str],
    timestamps: List[float],
    prompts: str,
    model_name: str,
    device: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    max_frames: int = MAX_FRAMES,
    max_objects: int = 20,
    near_thresh: float = 0.25,
) -> List[Dict]:
    if not prompts:
        return [{"error": "prompts_required"}]
    if len(frame_paths) > max_frames:
        step = max(len(frame_paths) // max_frames, 1)
        frame_paths = frame_paths[::step][:max_frames]
        timestamps = timestamps[::step][:max_frames]

    model, processor = _load_groundingdino(model_name, device)
    if model is None:
        return [{"error": "groundingdino_unavailable"}]

    try:
        import torch
    except Exception:
        return [{"error": "torch_unavailable"}]

    results: List[Dict] = []
    labels = [p.strip() for p in prompts.split(",") if p.strip()]
    if not labels:
        return [{"error": "prompts_required"}]

    for path, ts in zip(frame_paths, timestamps):
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue

        # 读取 BGR 版本用于颜色检测
        image_bgr = cv2.imread(path)
        img_w, img_h = image.size  # PIL: (W, H)

        inputs = processor(images=image, text=labels, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=device)
        # GroundingDINO processor API differs across versions.
        # Try with thresholds first, then fall back to older signature.
        try:
            detections = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            detections = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=target_sizes,
            )
        if not detections:
            continue
        det = detections[0]
        boxes = det.get("boxes", [])
        scores = det.get("scores", [])
        det_labels = det.get("labels", [])
        if boxes is None or len(boxes) == 0:
            continue

        nodes = []
        raw_boxes: List[List[float]] = []   # 保留用于计算关系
        for idx, (box, label, score) in enumerate(zip(boxes, det_labels, scores)):
            if idx >= max_objects:
                break
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            # label may be int id or already a string (newer processor)
            label_idx = None
            try:
                label_idx = int(label)
            except (TypeError, ValueError):
                label_idx = None
            if label_idx is not None:
                label_str = labels[label_idx] if label_idx < len(labels) else str(label_idx)
            else:
                label_str = str(label)

            box_list = [x1, y1, x2, y2]
            raw_boxes.append(box_list)

            node: Dict = {
                "id": idx,
                "label": label_str,
                "area": round(area, 2),
            }
            # 颜色检测
            if image_bgr is not None and _should_detect_color(label_str):
                color = _detect_color_for_box(image_bgr, box_list)
                if color:
                    node["color"] = color
            # 位置描述
            node["position"] = _box_to_position(box_list, float(img_w), float(img_h))
            nodes.append(node)

        relations = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                box_a = raw_boxes[i]
                box_b = raw_boxes[j]
                rel = _pair_relation(box_a, box_b)
                relations.append({"subj": nodes[i]["id"], "pred": rel, "obj": nodes[j]["id"]})

                diag_a = math.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])
                diag_b = math.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1])
                cx_a, cy_a = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
                cx_b, cy_b = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
                dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
                if dist <= near_thresh * max(diag_a, diag_b, 1.0):
                    relations.append({"subj": nodes[i]["id"], "pred": "near", "obj": nodes[j]["id"]})

                iou = _bbox_iou(box_a, box_b)
                if iou >= 0.3:
                    relations.append({"subj": nodes[i]["id"], "pred": "overlaps", "obj": nodes[j]["id"]})

        results.append({"time": round(float(ts), 2), "nodes": nodes, "relations": relations})
    return results
