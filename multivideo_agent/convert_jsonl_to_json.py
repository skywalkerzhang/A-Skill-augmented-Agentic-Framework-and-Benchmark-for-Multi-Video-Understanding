#!/usr/bin/env python3
"""
将 JSONL 格式的多视频数据转换为 eval_multivideo.py 需要的 JSON 格式
支持两种格式：
1. 标准格式：已有 metadata.reference_gt 和 metadata.options_gt
2. Counterfactual_Reasoning格式：从 reference 和 options 中提取视频路径
"""
import json
import sys
import re
from pathlib import Path

def extract_timestamps_from_filename(filename):
    """
    从文件名中提取时间戳
    例如: 80303_80338_80381.mp4 -> (80303, 80381) 或 (80303/30, 80381/30) 如果是以帧为单位
    尝试多种解析方式
    """
    # 移除扩展名
    base_name = filename.replace('.mp4', '').replace('.MP4', '')
    
    # 常见格式：<id>_<start>_<end>，其中 <id> 可能是非数字（如 YouTube id）
    # 我们优先取最后两个“纯数字”段作为 (start, end)。
    parts = base_name.split('_')
    nums = []
    for p in parts:
        if re.fullmatch(r"\d+(?:\.\d+)?", p):
            try:
                nums.append(float(p))
            except Exception:
                pass
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    
    # 如果无法解析，返回 None，让调用者使用默认值
    return None, None

def build_metadata_from_reference_options(raw_dic, dataset_folder, dataset_name=""):
    """
    从 Counterfactual_Reasoning 格式的 reference 和 options 构建 metadata
    dataset_name: 数据集名称（如 Counterfactual_Reasoning），用于构建完整路径
    """
    metadata = {
        "reference_gt": {},
        "options_gt": {}
    }
    
    def _to_abs_media_path(v: str) -> str:
        v = str(v)
        # If it already contains MultiVideo_Benchmark suffix, normalize it under dataset_folder.
        norm = v.replace("\\", "/")
        marker = "MultiVideo_Benchmark/"
        if marker in norm:
            suffix = norm.split(marker, 1)[1].lstrip("/")
            if dataset_folder:
                return f"{dataset_folder.rstrip('/')}/{suffix}"
            return v
        if v.startswith("/"):
            return v
        if dataset_folder and dataset_name:
            return f"{dataset_folder.rstrip('/')}/{dataset_name}/{v.lstrip('./')}"
        if dataset_folder:
            return f"{dataset_folder.rstrip('/')}/{v.lstrip('./')}"
        return v

    # 处理 reference（可能为 None）
    ref_data = raw_dic.get("reference")
    if isinstance(ref_data, dict) and ref_data.get("video"):
        ref_video = ref_data["video"]
        ref_file_path = _to_abs_media_path(ref_video)
        
        filename = Path(ref_video).name
        metadata["reference_gt"] = {
            "file_path": ref_file_path,
            "filename": filename
        }
        
        # 保存附加文本信息（如 counterfactual question/options）
        # 优先从 reference 字段读；如果没有，再从 raw metadata.reference（Counterfactual_Reasoning 常见）
        # 或 metadata.reference_gt 兜底（避免丢失 cf_* 文本）。
        if "question" in ref_data:
            metadata["reference_gt"]["cf_question"] = ref_data["question"]
        if "option1" in ref_data:
            metadata["reference_gt"]["cf_option1"] = ref_data["option1"]
        if "option2" in ref_data:
            metadata["reference_gt"]["cf_option2"] = ref_data["option2"]

        if (
            ("cf_question" not in metadata["reference_gt"])
            and ("cf_option1" not in metadata["reference_gt"])
            and ("cf_option2" not in metadata["reference_gt"])
        ):
            md = raw_dic.get("metadata")
            if isinstance(md, dict):
                # Counterfactual_Reasoning: metadata.reference carries per-video cf question/options
                md_ref = md.get("reference")
                if isinstance(md_ref, dict):
                    if md_ref.get("question"):
                        metadata["reference_gt"]["cf_question"] = md_ref.get("question")
                    if md_ref.get("option1"):
                        metadata["reference_gt"]["cf_option1"] = md_ref.get("option1")
                    if md_ref.get("option2"):
                        metadata["reference_gt"]["cf_option2"] = md_ref.get("option2")

                ref_gt = md.get("reference_gt")
                if isinstance(ref_gt, dict):
                    for k in ("cf_question", "cf_option1", "cf_option2"):
                        if k in ref_gt and ref_gt.get(k):
                            metadata["reference_gt"][k] = ref_gt.get(k)
    
    # 处理 options
    options = raw_dic.get("options")
    if isinstance(options, list):
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if "label" in opt and "video" in opt:
                label = opt["label"]
                opt_video = opt["video"]
                opt_file_path = _to_abs_media_path(opt_video)
                
                filename = Path(opt_video).name
                metadata["options_gt"][label] = {
                    "file_path": opt_file_path,
                    "filename": filename
                }
                
                # 保存附加文本信息（如 counterfactual question/options）
                # Candidate question may be stored as "question" or normalized to "text".
                cf_q = opt.get("question") or opt.get("text")
                if cf_q:
                    metadata["options_gt"][label]["cf_question"] = cf_q
                if "option1" in opt:
                    metadata["options_gt"][label]["cf_option1"] = opt["option1"]
                if "option2" in opt:
                    metadata["options_gt"][label]["cf_option2"] = opt["option2"]
    
    return metadata

def convert_jsonl_to_json(input_jsonl_path, output_json_path, dataset_folder="", dataset_name=""):
    """
    转换 JSONL 到 JSON，只保留必要字段
    支持两种数据格式：
    1. 已有metadata格式（Action_Matching等）
    2. Counterfactual_Reasoning格式（需要从reference/options构建metadata）
    
    dataset_name: 数据集名称（如 Counterfactual_Reasoning），用于构建完整路径
    """
    data_list = []
    
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                raw_dic = json.loads(line.strip())

                # ===== 关键策略：永远不要把原始 metadata（可能含 gt_*）带到输出里 =====
                # 统一优先用 reference/options 重新构造“干净版” metadata（只含 file_path/filename/start/end + cf_* 文本）
                options = raw_dic.get("options")
                metadata = None
                if isinstance(options, list):
                    metadata = build_metadata_from_reference_options(raw_dic, dataset_folder, dataset_name)

                if metadata and metadata.get("reference_gt"):
                    converted_dic = {
                        "question": raw_dic.get("question", ""),
                        "answer": raw_dic.get("answer", ""),
                        "metadata": metadata,
                    }
                elif isinstance(options, list):
                    # no-ref：reference 为空时，走 MVU/no_ref 格式（video_paths）
                    video_paths = []
                    for opt in options:
                        if not isinstance(opt, dict):
                            continue
                        v = opt.get("video")
                        if not v:
                            continue
                        v = str(v)
                        if v.startswith("/"):
                            abs_v = v
                        else:
                            # Only strip leading "./" (do NOT destroy "../" prefixes).
                            while v.startswith("./"):
                                v = v[2:]
                            # If path contains MultiVideo_Benchmark suffix, normalize under dataset_folder.
                            norm = v.replace("\\", "/")
                            marker = "MultiVideo_Benchmark/"
                            if marker in norm and dataset_folder:
                                suffix = norm.split(marker, 1)[1].lstrip("/")
                                abs_v = f"{dataset_folder.rstrip('/')}/{suffix}"
                            elif dataset_folder:
                                abs_v = f"{dataset_folder.rstrip('/')}/{v}"
                            else:
                                abs_v = v
                        video_paths.append(abs_v)

                    if not video_paths:
                        print(f"Warning: options 中找不到 video 字段，跳过", file=sys.stderr)
                        continue

                    converted_dic = {
                        "question": raw_dic.get("question", ""),
                        "answer": raw_dic.get("answer", ""),
                        "video_paths": video_paths,
                    }
                else:
                    print(f"Warning: 无法识别的数据格式，跳过", file=sys.stderr)
                    continue
                
                # 如果有 task 字段，也保留（虽然可能不需要）
                if "task" in raw_dic:
                    converted_dic["task"] = raw_dic["task"]
                
                data_list.append(converted_dic)
                
            except json.JSONDecodeError as e:
                print(f"Warning: 跳过无效的 JSON 行: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: 处理数据时出错: {e}", file=sys.stderr)
                continue
    
    # 写入 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 转换完成: {len(data_list)} 条数据")
    print(f"   输入: {input_jsonl_path}")
    print(f"   输出: {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert_jsonl_to_json.py <input.jsonl> <output.json> [dataset_folder] [dataset_name]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    dataset_folder = sys.argv[3] if len(sys.argv) > 3 else ""
    dataset_name = sys.argv[4] if len(sys.argv) > 4 else ""
    
    if not Path(input_path).exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)
    
    convert_jsonl_to_json(input_path, output_path, dataset_folder, dataset_name)


