import base64
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageFile

from .video_utils_base import MAX_RESOLUTION
from .video_utils_frames import extract_uniform_frames


def is_valid_frame(frame_path):
    if not os.path.exists(frame_path):
        return False
    try:
        with Image.open(frame_path) as img:
            img.verify()
        with Image.open(frame_path) as img:
            img.load()
        return True
    except Exception:
        return False


def load_image(image_path, max_dimension_of_height_width=768):
    try:
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return None

        frame = None
        try:
            pil_img = Image.open(image_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            frame = np.array(pil_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PIL读取失败 {image_path}: {e}")

        if frame is None:
            try:
                frame = cv2.imread(image_path)
                if frame is not None:
                    print(f"使用OpenCV成功读取: {image_path}")
            except Exception as e:
                print(f"OpenCV读取失败 {image_path}: {e}")

        if frame is None:
            try:
                with open(image_path, "rb") as f:
                    img_data = np.frombuffer(f.read(), np.uint8)
                    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if frame is not None:
                    print(f"使用二进制读取成功: {image_path}")
            except Exception as e:
                print(f"二进制读取失败 {image_path}: {e}")
                return None

        if frame is None:
            raise ValueError(f"无法读取图像: {image_path}")

        if frame.size == 0:
            raise ValueError(f"读取的图像为空: {image_path}")

        height, width = frame.shape[:2]
        if max(height, width) > max_dimension_of_height_width:
            max_dimension = max_dimension_of_height_width
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized_frame = frame

        success, buffer = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            print(f"图片编码失败: {image_path}")
            return None

        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return None


def fix_corrupted_images(frame_folder):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    png_files = glob.glob(os.path.join(frame_folder, "*.png"))
    for png_file in png_files:
        try:
            with Image.open(png_file) as img:
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(png_file, "PNG", optimize=True)
                print(f"修复成功: {png_file}")
        except Exception as e:
            print(f"无法修复: {png_file}, 错误: {e}")
            try:
                os.remove(png_file)
                print(f"删除损坏文件: {png_file}")
            except Exception:
                print(f"无法删除损坏文件: {png_file}")


def regenerate_frames_if_corrupted(video_path, frame_folder, max_frames=32):
    corrupted_files = []
    png_files = glob.glob(os.path.join(frame_folder, "*.png"))

    for png_file in png_files:
        try:
            with Image.open(png_file) as img:
                img.load()
        except Exception as e:
            print(f"发现损坏文件: {png_file}, 错误: {e}")
            corrupted_files.append(png_file)

    if corrupted_files:
        print(f"发现 {len(corrupted_files)} 个损坏文件，重新提取帧...")
        for png_file in png_files:
            try:
                os.remove(png_file)
            except Exception:
                pass
        return extract_uniform_frames(video_path, frame_folder, max_frames)

    return None, None


def image_paths_to_base64(image_paths, max_resolution=MAX_RESOLUTION):
    base64Frames = []
    for image_path in image_paths:
        image_path = image_path.strip()
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}")
            return False

        height, width = frame.shape[:2]
        max_dimension = max_resolution

        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", resized_frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    return base64Frames
