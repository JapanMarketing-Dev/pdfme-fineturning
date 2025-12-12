#!/usr/bin/env python3
"""
PDFme Form Field Detection - 評価スクリプト
==========================================
モデルの精度を正しく評価します。
座標系を統一してIoU、Recall、Precisionを計算。
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


# プロンプトテンプレート
SYSTEM_PROMPT = """あなたは日本の書類（申請書、届出書など）を分析するエキスパートです。
書類には2種類の入力欄があります：
1. 担当者（申請者・顧客）が記入する欄 → 検出対象
2. 職員（役所・会社の担当者）が記入する欄 → 対象外

担当者が記入する欄の例：氏名、住所、電話番号、生年月日、メールアドレス、署名欄など
職員が記入する欄の例：受付番号、処理日、担当印、審査欄、決裁欄など"""

USER_PROMPT = """この画像から、担当者（申請者・顧客）が記入する入力フィールドの位置をすべて検出してください。
職員が記入する欄は除外してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""


def normalize_bbox_to_1000(bbox: list, img_width: int, img_height: int) -> list:
    """ピクセル座標を0-1000正規化座標に変換"""
    x1, y1, x2, y2 = bbox
    return [
        int((x1 / img_width) * 1000),
        int((y1 / img_height) * 1000),
        int((x2 / img_width) * 1000),
        int((y2 / img_height) * 1000),
    ]


def denormalize_bbox_from_1000(bbox: list, img_width: int, img_height: int) -> list:
    """0-1000正規化座標をピクセル座標に変換"""
    x1, y1, x2, y2 = bbox
    return [
        int((x1 / 1000) * img_width),
        int((y1 / 1000) * img_height),
        int((x2 / 1000) * img_width),
        int((y2 / 1000) * img_height),
    ]


def calculate_iou(box1: list, box2: list) -> float:
    """
    2つのbboxのIoU（Intersection over Union）を計算
    bbox形式: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 交差領域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 交差領域の面積（負の値は0にクリップ）
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    
    # 各boxの面積
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Union
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def evaluate_predictions(
    predictions: list[list],
    ground_truths: list[list],
    iou_threshold: float = 0.5
) -> dict:
    """
    予測と正解のIoU、Recall、Precisionを計算
    
    Args:
        predictions: 予測bboxリスト（正規化座標）
        ground_truths: 正解bboxリスト（正規化座標）
        iou_threshold: マッチ判定のIoU閾値
    
    Returns:
        評価結果の辞書
    """
    if not predictions and not ground_truths:
        return {"recall": 1.0, "precision": 1.0, "avg_iou": 1.0, "matched": 0, "total_gt": 0, "total_pred": 0}
    
    if not predictions:
        return {"recall": 0.0, "precision": 0.0, "avg_iou": 0.0, "matched": 0, "total_gt": len(ground_truths), "total_pred": 0}
    
    if not ground_truths:
        return {"recall": 0.0, "precision": 0.0, "avg_iou": 0.0, "matched": 0, "total_gt": 0, "total_pred": len(predictions)}
    
    # 各正解に対して最もIoUの高い予測を見つける
    all_ious = []
    matched_gt = set()
    matched_pred = set()
    
    for gt_idx, gt_box in enumerate(ground_truths):
        best_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx, pred_box in enumerate(predictions):
            iou = calculate_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        
        all_ious.append(best_iou)
        
        if best_iou >= iou_threshold and best_pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
    
    num_matched = len(matched_gt)
    recall = num_matched / len(ground_truths) if ground_truths else 0.0
    precision = num_matched / len(predictions) if predictions else 0.0
    avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    
    return {
        "recall": recall,
        "precision": precision,
        "avg_iou": avg_iou,
        "matched": num_matched,
        "total_gt": len(ground_truths),
        "total_pred": len(predictions),
        "ious": all_ious,
    }


def parse_model_output(output: str) -> list[list]:
    """モデル出力からbboxリストを抽出（複数の形式に対応）"""
    bboxes = []
    
    # コードブロックを除去
    clean_output = output
    if "```json" in clean_output:
        clean_output = clean_output.split("```json")[-1]
    if "```" in clean_output:
        clean_output = clean_output.split("```")[0]
    clean_output = clean_output.strip()
    
    # 方法1: 正規のJSON配列パース
    try:
        json_start = clean_output.find("[")
        json_end = clean_output.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_str = clean_output[json_start:json_end]
            parsed = json.loads(json_str)
            
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if ("bbox" in key.lower()) and isinstance(value, list) and len(value) == 4:
                                bboxes.append(value)
                    elif isinstance(item, list) and len(item) == 4:
                        bboxes.append(item)
            if bboxes:
                return bboxes
    except json.JSONDecodeError:
        pass
    
    # 方法2: 途切れたJSONでも個別オブジェクトを抽出
    import re
    # {"bbox_XXX": [x1, y1, x2, y2], ...} パターンを検索
    pattern = r'\{[^}]*"bbox[^"]*"\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\][^}]*\}'
    matches = re.findall(pattern, clean_output, re.IGNORECASE)
    for match in matches:
        try:
            bbox = [int(match[0]), int(match[1]), int(match[2]), int(match[3])]
            bboxes.append(bbox)
        except (ValueError, IndexError):
            pass
    
    if bboxes:
        return bboxes
    
    # 方法3: オブジェクト形式 {...}
    try:
        json_start = clean_output.find("{")
        json_end = clean_output.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = clean_output[json_start:json_end]
            parsed = json.loads(json_str)
            
            if "applicant_fields" in parsed:
                for field in parsed["applicant_fields"]:
                    if "bbox" in field and isinstance(field["bbox"], list) and len(field["bbox"]) == 4:
                        bboxes.append(field["bbox"])
            
            for key, value in parsed.items():
                if ("bbox" in key.lower()) and isinstance(value, list) and len(value) == 4:
                    bboxes.append(value)
    except json.JSONDecodeError:
        pass
    
    return bboxes


def load_model(model_path: str, base_model: str, use_4bit: bool = True):
    """モデルをロード"""
    hf_token = os.environ.get("HF_TOKEN")
    
    print(f"ベースモデル: {base_model}")
    print(f"LoRAアダプター: {model_path if model_path else 'なし'}")
    
    # プロセッサをロード
    processor = AutoProcessor.from_pretrained(
        base_model,
        token=hf_token,
        trust_remote_code=True,
    )
    
    # モデルをロード
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
    
    # LoRAアダプターをロード（指定された場合）
    if model_path:
        print(f"LoRAアダプターをロード: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    
    model.eval()
    return model, processor


def predict_single(model, processor, image: Image.Image) -> str:
    """単一画像に対して推論を実行"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response


def main():
    parser = argparse.ArgumentParser(description="PDFme Form Field Detection 評価")
    parser.add_argument("--lora", default=None, help="LoRAアダプターパス（なしの場合はベースモデルのみ）")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct", help="ベースモデル")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="4bit量子化を使用")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU閾値")
    parser.add_argument("--output", default=None, help="結果をJSONで保存")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PDFme Form Field Detection - 評価")
    print("=" * 60)
    
    # データセットをロード
    print("\nデータセットをロード中...")
    ds = load_dataset("hand-dot/pdfme-form-field-dataset")
    print(f"データ数: {len(ds['train'])}件")
    
    # モデルをロード
    print("\nモデルをロード中...")
    model, processor = load_model(args.lora, args.base_model, args.use_4bit)
    print("モデルロード完了")
    
    # 評価実行
    print(f"\n評価開始（IoU閾値: {args.iou_threshold}）...")
    
    all_results = []
    total_recall = 0.0
    total_precision = 0.0
    total_avg_iou = 0.0
    total_matched = 0
    total_gt_count = 0
    total_pred_count = 0
    
    for i, item in enumerate(tqdm(ds["train"], desc="評価")):
        img = item["image"]
        gt_bboxes_pixel = item["bboxes"]  # ピクセル座標
        img_width, img_height = img.size
        
        # 正解をピクセル座標から0-1000正規化座標に変換
        gt_bboxes_norm = [
            normalize_bbox_to_1000(bbox, img_width, img_height)
            for bbox in gt_bboxes_pixel
        ]
        
        # 推論実行
        output = predict_single(model, processor, img)
        
        # 出力をパース（モデルは0-1000正規化座標を出力）
        pred_bboxes_norm = parse_model_output(output)
        
        # 評価（両方とも0-1000正規化座標で比較）
        result = evaluate_predictions(pred_bboxes_norm, gt_bboxes_norm, args.iou_threshold)
        
        sample_result = {
            "sample_idx": i,
            "image_size": [img_width, img_height],
            "gt_count": len(gt_bboxes_norm),
            "pred_count": len(pred_bboxes_norm),
            "recall": result["recall"],
            "precision": result["precision"],
            "avg_iou": result["avg_iou"],
            "matched": result["matched"],
        }
        all_results.append(sample_result)
        
        total_matched += result["matched"]
        total_gt_count += result["total_gt"]
        total_pred_count += result["total_pred"]
        total_avg_iou += result["avg_iou"] * result["total_gt"]
        
        if args.verbose:
            print(f"\n--- サンプル {i} ---")
            print(f"画像サイズ: {img_width}x{img_height}")
            print(f"正解bbox数: {len(gt_bboxes_norm)}")
            print(f"予測bbox数: {len(pred_bboxes_norm)}")
            print(f"Recall: {result['recall']:.2%}")
            print(f"Precision: {result['precision']:.2%}")
            print(f"平均IoU: {result['avg_iou']:.4f}")
            print(f"モデル出力（一部）: {output[:200]}...")
    
    # 全体の結果を計算
    overall_recall = total_matched / total_gt_count if total_gt_count > 0 else 0.0
    overall_precision = total_matched / total_pred_count if total_pred_count > 0 else 0.0
    overall_avg_iou = total_avg_iou / total_gt_count if total_gt_count > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("評価結果")
    print("=" * 60)
    print(f"IoU閾値: {args.iou_threshold}")
    print(f"サンプル数: {len(ds['train'])}")
    print(f"総正解bbox数: {total_gt_count}")
    print(f"総予測bbox数: {total_pred_count}")
    print(f"マッチ数: {total_matched}")
    print("-" * 40)
    print(f"Recall: {overall_recall:.2%}")
    print(f"Precision: {overall_precision:.2%}")
    print(f"平均IoU: {overall_avg_iou:.4f}")
    print("=" * 60)
    
    # 結果を保存
    if args.output:
        output_data = {
            "config": {
                "lora": args.lora,
                "base_model": args.base_model,
                "iou_threshold": args.iou_threshold,
            },
            "summary": {
                "recall": overall_recall,
                "precision": overall_precision,
                "avg_iou": overall_avg_iou,
                "total_gt": total_gt_count,
                "total_pred": total_pred_count,
                "total_matched": total_matched,
            },
            "samples": all_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n結果を保存: {args.output}")


if __name__ == "__main__":
    main()

