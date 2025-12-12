#!/usr/bin/env python3
"""
PDFme Form Field Detection - データ拡張スクリプト
================================================
学習データを拡張して、モデルの汎化性能を向上させます。

拡張手法:
1. 回転（±5度）
2. スケール変換（90%-110%）
3. 明るさ/コントラスト調整
4. 左右反転（オプション）
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageEnhance
from datasets import load_dataset, Dataset
from tqdm import tqdm
import math


def rotate_bbox(bbox: list, angle: float, img_width: int, img_height: int) -> list:
    """
    bboxを画像中心を基準に回転
    
    Args:
        bbox: [x1, y1, x2, y2] ピクセル座標
        angle: 回転角度（度）
        img_width, img_height: 画像サイズ
    
    Returns:
        回転後のbbox [x1, y1, x2, y2]
    """
    # 画像中心
    cx, cy = img_width / 2, img_height / 2
    
    # ラジアンに変換
    rad = math.radians(-angle)  # PILは反時計回りが正
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    x1, y1, x2, y2 = bbox
    
    # 4つの角を回転
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    rotated_corners = []
    
    for x, y in corners:
        # 中心を原点に移動
        x_centered = x - cx
        y_centered = y - cy
        
        # 回転
        x_rot = x_centered * cos_a - y_centered * sin_a
        y_rot = x_centered * sin_a + y_centered * cos_a
        
        # 中心を戻す
        rotated_corners.append((x_rot + cx, y_rot + cy))
    
    # 回転後のbounding boxを計算
    xs = [p[0] for p in rotated_corners]
    ys = [p[1] for p in rotated_corners]
    
    new_x1 = max(0, min(xs))
    new_y1 = max(0, min(ys))
    new_x2 = min(img_width, max(xs))
    new_y2 = min(img_height, max(ys))
    
    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def scale_bbox(bbox: list, scale: float, img_width: int, img_height: int) -> list:
    """
    bboxをスケール変換（画像中心基準）
    """
    cx, cy = img_width / 2, img_height / 2
    x1, y1, x2, y2 = bbox
    
    # 中心からの相対位置をスケール
    new_x1 = cx + (x1 - cx) * scale
    new_y1 = cy + (y1 - cy) * scale
    new_x2 = cx + (x2 - cx) * scale
    new_y2 = cy + (y2 - cy) * scale
    
    # 画像範囲内にクリップ
    new_x1 = max(0, min(img_width, new_x1))
    new_y1 = max(0, min(img_height, new_y1))
    new_x2 = max(0, min(img_width, new_x2))
    new_y2 = max(0, min(img_height, new_y2))
    
    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def flip_bbox_horizontal(bbox: list, img_width: int) -> list:
    """水平反転"""
    x1, y1, x2, y2 = bbox
    return [img_width - x2, y1, img_width - x1, y2]


def augment_image(
    image: Image.Image,
    bboxes: list,
    augmentation_type: str
) -> tuple[Image.Image, list]:
    """
    画像とbboxを拡張
    
    Args:
        image: 元画像
        bboxes: bboxリスト
        augmentation_type: 拡張タイプ
    
    Returns:
        (拡張後画像, 拡張後bboxリスト)
    """
    img_width, img_height = image.size
    new_image = image.copy()
    new_bboxes = [bbox.copy() for bbox in bboxes]
    
    if augmentation_type == "rotate_cw":
        # 時計回り5度回転
        angle = 5
        new_image = image.rotate(-angle, expand=False, fillcolor=(255, 255, 255))
        new_bboxes = [rotate_bbox(bbox, angle, img_width, img_height) for bbox in bboxes]
    
    elif augmentation_type == "rotate_ccw":
        # 反時計回り5度回転
        angle = -5
        new_image = image.rotate(-angle, expand=False, fillcolor=(255, 255, 255))
        new_bboxes = [rotate_bbox(bbox, angle, img_width, img_height) for bbox in bboxes]
    
    elif augmentation_type == "scale_up":
        # 110%にスケールアップ（中央部分を使用）
        scale = 1.1
        new_w, new_h = int(img_width * scale), int(img_height * scale)
        scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 中央をクロップ
        left = (new_w - img_width) // 2
        top = (new_h - img_height) // 2
        new_image = scaled.crop((left, top, left + img_width, top + img_height))
        
        # bboxを調整（スケールアップ後、クロップ分をオフセット）
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            new_x1 = int(x1 * scale) - left
            new_y1 = int(y1 * scale) - top
            new_x2 = int(x2 * scale) - left
            new_y2 = int(y2 * scale) - top
            
            # 画像範囲内にクリップ
            new_x1 = max(0, min(img_width, new_x1))
            new_y1 = max(0, min(img_height, new_y1))
            new_x2 = max(0, min(img_width, new_x2))
            new_y2 = max(0, min(img_height, new_y2))
            
            # 有効なbboxのみ追加
            if new_x2 > new_x1 and new_y2 > new_y1:
                new_bboxes.append([new_x1, new_y1, new_x2, new_y2])
    
    elif augmentation_type == "scale_down":
        # 90%にスケールダウン（パディング追加）
        scale = 0.9
        new_w, new_h = int(img_width * scale), int(img_height * scale)
        scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 白背景でパディング
        new_image = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        left = (img_width - new_w) // 2
        top = (img_height - new_h) // 2
        new_image.paste(scaled, (left, top))
        
        # bboxを調整
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            new_x1 = int(x1 * scale) + left
            new_y1 = int(y1 * scale) + top
            new_x2 = int(x2 * scale) + left
            new_y2 = int(y2 * scale) + top
            new_bboxes.append([new_x1, new_y1, new_x2, new_y2])
    
    elif augmentation_type == "brightness_up":
        # 明るさ+10%
        enhancer = ImageEnhance.Brightness(image)
        new_image = enhancer.enhance(1.1)
        new_bboxes = [bbox.copy() for bbox in bboxes]
    
    elif augmentation_type == "brightness_down":
        # 明るさ-10%
        enhancer = ImageEnhance.Brightness(image)
        new_image = enhancer.enhance(0.9)
        new_bboxes = [bbox.copy() for bbox in bboxes]
    
    elif augmentation_type == "contrast_up":
        # コントラスト+20%
        enhancer = ImageEnhance.Contrast(image)
        new_image = enhancer.enhance(1.2)
        new_bboxes = [bbox.copy() for bbox in bboxes]
    
    elif augmentation_type == "contrast_down":
        # コントラスト-20%
        enhancer = ImageEnhance.Contrast(image)
        new_image = enhancer.enhance(0.8)
        new_bboxes = [bbox.copy() for bbox in bboxes]
    
    return new_image, new_bboxes


def main():
    parser = argparse.ArgumentParser(description="PDFme データ拡張")
    parser.add_argument("--output-dir", default="data/augmented", help="出力ディレクトリ")
    parser.add_argument("--augmentations", default="rotate_cw,rotate_ccw,scale_up,scale_down,brightness_up,brightness_down,contrast_up,contrast_down",
                        help="適用する拡張（カンマ区切り）")
    parser.add_argument("--save-images", action="store_true", help="画像をファイルに保存")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PDFme Form Field Detection - データ拡張")
    print("=" * 60)
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データセットをロード
    print("\nデータセットをロード中...")
    ds = load_dataset("hand-dot/pdfme-form-field-dataset")
    original_count = len(ds["train"])
    print(f"元データ数: {original_count}件")
    
    # 拡張タイプ
    augmentation_types = args.augmentations.split(",")
    print(f"拡張タイプ: {augmentation_types}")
    print(f"拡張後の予想データ数: {original_count * (1 + len(augmentation_types))}件")
    
    # 拡張データを生成
    augmented_samples = []
    
    # 元データをそのまま追加
    for i, item in enumerate(ds["train"]):
        augmented_samples.append({
            "image": item["image"],
            "bboxes": item["bboxes"],
            "augmentation": "original",
            "original_idx": i,
        })
    
    # 拡張データを追加
    for aug_type in tqdm(augmentation_types, desc="拡張タイプ"):
        for i, item in enumerate(ds["train"]):
            image = item["image"]
            bboxes = item["bboxes"]
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # 拡張を適用
            aug_image, aug_bboxes = augment_image(image, bboxes, aug_type)
            
            augmented_samples.append({
                "image": aug_image,
                "bboxes": aug_bboxes,
                "augmentation": aug_type,
                "original_idx": i,
            })
    
    print(f"\n拡張後のデータ数: {len(augmented_samples)}件")
    
    # 画像を保存（オプション）
    if args.save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(tqdm(augmented_samples, desc="画像保存")):
            img_path = images_dir / f"sample_{i:04d}_{sample['augmentation']}.png"
            sample["image"].save(img_path)
    
    # データセットをJSON形式で保存（画像パスと共に）
    metadata = {
        "original_count": original_count,
        "augmented_count": len(augmented_samples),
        "augmentation_types": augmentation_types,
        "created_at": datetime.now().isoformat(),
        "samples": [],
    }
    
    for i, sample in enumerate(augmented_samples):
        metadata["samples"].append({
            "idx": i,
            "augmentation": sample["augmentation"],
            "original_idx": sample["original_idx"],
            "bboxes": sample["bboxes"],
            "image_size": list(sample["image"].size),
        })
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nメタデータを保存: {metadata_path}")
    
    # HuggingFace Dataset形式で保存
    hf_dataset = Dataset.from_list([
        {"image": s["image"], "bboxes": s["bboxes"]}
        for s in augmented_samples
    ])
    
    dataset_path = output_dir / "dataset"
    hf_dataset.save_to_disk(str(dataset_path))
    print(f"データセットを保存: {dataset_path}")
    
    print("\n" + "=" * 60)
    print("データ拡張完了!")
    print(f"元データ: {original_count}件 → 拡張後: {len(augmented_samples)}件")
    print("=" * 60)
    
    return augmented_samples


if __name__ == "__main__":
    main()

