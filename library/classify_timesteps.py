import os
import sys
import argparse
import shutil
import math
import json
import traceback
from PIL import Image
import numpy as np

# 支援的圖片格式
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def get_images_in_dir(directory):
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and is_image(f)]

def write_progress(job_dir, current, total, status):
    if not job_dir:
        return
    try:
        progress_path = os.path.join(job_dir, "split_progress.json")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({
                "current": current,
                "total": total,
                "status": status
            }, f)
    except Exception as e:
        print(f"Error writing progress: {e}")

def classify_images_in_folder(folder_path, trigger_word, dest_parent_dir, job_dir=None, orig_repeats=1):
    images = get_images_in_dir(folder_path)
    if not images:
        print(f"No images found in {folder_path}, skipping.")
        write_progress(job_dir, 0, 0, "No images found")
        return

    total = len(images)
    print(f"Processing {total} images in {folder_path}...")
    write_progress(job_dir, 0, total, "Initializing mask model...")
    
    # 嘗試使用 transparent-background 進行遮罩計算
    remover = None
    try:
        from transparent_background import Remover
        remover = Remover(device="cuda", mode="base-nightly")
        print("Successfully initialized transparent-background Remover on CUDA.")
    except Exception as e:
        try:
            from transparent_background import Remover
            remover = Remover(device="cpu", mode="base-nightly")
            print("Successfully initialized transparent-background Remover on CPU.")
        except Exception as e2:
            print(f"Could not load transparent-background Remover (Error: {e2}). Falling back to file size sorting.")

    weights = []
    
    # 計算每張圖片的權重
    for idx, img_path in enumerate(images):
        write_progress(job_dir, idx, total, f"Calculating mask: {idx}/{total}")
        weight = 0.0
        if remover is not None:
            try:
                img = Image.open(img_path)
                # 優先嘗試讀取圖片本身的 Alpha 通道（如果有）
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    alpha = img.convert('RGBA').split()[-1]
                    mask_arr = np.array(alpha)
                    weight = float(np.mean(mask_arr) / 255.0)
                else:
                    # 使用模型去背景
                    rgb_img = img.convert("RGB")
                    mask = remover.process(rgb_img, type="map")
                    if not isinstance(mask, Image.Image):
                        mask = Image.fromarray(np.asarray(mask))
                    mask_arr = np.array(mask.convert("L"))
                    weight = float(np.mean(mask_arr) / 255.0)
            except Exception as ex:
                print(f"Error processing image {img_path} with mask model: {ex}. Using file size fallback.")
                weight = float(os.path.getsize(img_path))
        else:
            # Fallback 1: 優先嘗試讀取圖片本身的 Alpha 通道（如果有）
            try:
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    alpha = img.convert('RGBA').split()[-1]
                    weight = float(np.mean(np.array(alpha)) / 255.0)
                else:
                    # Fallback 2: 使用檔案大小作為排序依據
                    weight = float(os.path.getsize(img_path))
            except Exception:
                weight = float(os.path.getsize(img_path))
                
        weights.append((img_path, weight))

    # 按權重從大到小排序
    weights.sort(key=lambda x: x[1], reverse=True)
    
    # 決定分界點 (前 50% 到 high，後 50% 到 low)
    split_idx = math.ceil(len(weights) / 2)
    high_images = [x[0] for x in weights[:split_idx]]
    low_images = [x[0] for x in weights[split_idx:]]

    write_progress(job_dir, total, total, "Moving files to target directories...")

    # 建立目標子目錄
    suggested_high_dir = os.path.join(dest_parent_dir, f"1_{trigger_word} {{Suggested High}}")
    suggested_low_dir = os.path.join(dest_parent_dir, f"1_{trigger_word} {{Suggested Low}}")
    mid_dir = os.path.join(dest_parent_dir, f"1_{trigger_word} {{MId}}")
    low_dir = os.path.join(dest_parent_dir, f"1_{trigger_word} {{Low}}")
    high_dir = os.path.join(dest_parent_dir, f"1_{trigger_word} {{High}}")

    for d in [suggested_high_dir, suggested_low_dir, mid_dir, low_dir, high_dir]:
        os.makedirs(d, exist_ok=True)

    # 移動圖片
    for img_path in high_images:
        shutil.move(img_path, os.path.join(suggested_high_dir, os.path.basename(img_path)))
        # 同時移動可能的對應 txt 標籤檔
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            shutil.move(txt_path, os.path.join(suggested_high_dir, os.path.basename(txt_path)))

    for img_path in low_images:
        shutil.move(img_path, os.path.join(suggested_low_dir, os.path.basename(img_path)))
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            shutil.move(txt_path, os.path.join(suggested_low_dir, os.path.basename(txt_path)))

    write_progress(job_dir, total, total, "done")
    print(f"Successfully classified: {len(high_images)} to {suggested_high_dir}, {len(low_images)} to {suggested_low_dir}")

def main():
    parser = argparse.ArgumentParser(description="Auto split timesteps using automask pixel classification.")
    parser.add_argument("--image_dir", required=True, help="Image directory path")
    parser.add_argument("--trigger_word", default="miku", help="Default trigger word if parsing fails")
    parser.add_argument("--batch_import", action="store_true", help="Batch import mode")
    parser.add_argument("--job_dir", default="", help="Job directory to store split progress")
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist.")
        sys.exit(1)

    if args.batch_import:
        # 遍歷第一層資料夾
        folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
        for folder_name in folders:
            folder_path = os.path.join(image_dir, folder_name)
            
            # 解析資料夾 repeats 和 trigger_word
            parts = folder_name.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                repeats = int(parts[0])
                trigger_word = parts[1].strip()
            else:
                repeats = 1
                trigger_word = folder_name.strip()

            # 排除已是分類後的子資料夾
            if "{" in trigger_word or "}" in trigger_word:
                print(f"Skipping already classified folder: {folder_name}")
                continue

            classify_images_in_folder(folder_path, trigger_word, image_dir, args.job_dir, repeats)
            
            # 清理已清空的資料夾
            try:
                if not get_images_in_dir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Removed emptied folder: {folder_path}")
            except Exception as e:
                print(f"Failed to remove folder {folder_path}: {e}")
    else:
        # 單一資料夾模式
        folder_name = os.path.basename(image_dir)
        trigger_word = args.trigger_word if args.trigger_word else folder_name
        classify_images_in_folder(image_dir, trigger_word, image_dir, args.job_dir, 1)

if __name__ == "__main__":
    main()
