#!/usr/bin/env python3
"""
PDFme Form Field Dataset - ダウンロードと確認スクリプト
======================================================
HuggingFaceからデータセットをダウンロードし、内容を確認します。
"""

import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main():
    print("=" * 60)
    print("PDFme Form Field Dataset - ダウンロードと確認")
    print("=" * 60)
    print("注意: このデータセットのbboxesは")
    print("      「担当者（申請者）が記入するフィールド」のみを含みます")
    print("=" * 60)
    
    # データセットをダウンロード
    print("\nデータセットをダウンロード中...")
    ds = load_dataset("hand-dot/pdfme-form-field-dataset")
    
    print("\n=== データセット情報 ===")
    print(ds)
    
    print("\n=== カラム ===")
    print(ds["train"].column_names)
    
    print(f"\n=== サンプル数 ===")
    print(f"Train: {len(ds['train'])}件")
    
    print("\n=== 全サンプルの詳細 ===")
    for i, item in enumerate(ds["train"]):
        img = item["image"]
        bboxes = item["bboxes"]
        print(f"  Sample {i}:")
        print(f"    画像サイズ: {img.size}")
        print(f"    担当者用フィールド数: {len(bboxes)}")
        if bboxes:
            print(f"    bboxes例: {bboxes[0]}")
    
    # 画像を保存（オプション）
    output_dir = Path("data/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== 画像を {output_dir} に保存中... ===")
    for i, item in enumerate(ds["train"]):
        img = item["image"]
        img_path = output_dir / f"sample_{i:03d}.png"
        img.save(img_path)
        print(f"  保存: {img_path}")
    
    print("\n完了!")
    return ds


if __name__ == "__main__":
    main()

