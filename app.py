import gradio as gr
import json
import re
import torch
import traceback
from PIL import Image, ImageDraw

try:
    import spaces
    GPU_DECORATOR = spaces.GPU(duration=120)  # 2分
except ImportError:
    def GPU_DECORATOR(fn):
        return fn

model = None
processor = None

def load_model():
    global model, processor
    if model is not None:
        return
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    
    # 32Bモデルに変更
    BASE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
    LORA_ADAPTER = "takumi123xxx/pdfme-form-field-detector-lora-32b"
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print("Loading model (32B, this may take a while)...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"4bit failed: {e}, trying fp16...")
        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_ADAPTER)
    model.eval()
    print("Model loaded!")

def draw_boxes(image, bboxes):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
    
    w, h = img.size
    for i, box in enumerate(bboxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = box["bbox"]
        x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
        x2, y2 = max(0, min(x2, w-1)), max(0, min(y2, h-1))
        
        if x2 > x1 and y2 > y1:
            for j in range(3):
                draw.rectangle([x1-j, y1-j, x2+j, y2+j], outline=color)
            label_y = max(0, y1-18)
            draw.rectangle([x1, label_y, x1+80, label_y+18], fill=color)
            draw.text((x1+2, label_y+2), f"{i+1}", fill="white")
    return img

def parse_output(text, w, h):
    bboxes = []
    try:
        json_match = re.search(r'[\[\{].*[\]\}]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "applicant_fields" in data:
                for f in data["applicant_fields"]:
                    if "bbox" in f and len(f["bbox"]) >= 4:
                        b = f["bbox"]
                        bboxes.append({
                            "bbox": [int(b[0]/1000*w), int(b[1]/1000*h), int(b[2]/1000*w), int(b[3]/1000*h)],
                            "label": str(f.get("label", "field"))
                        })
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key in ["bbox", "bbox_2d", "bbox_0100", "bbox_01000", "bbox_0_to_1000"]:
                            if key in item and isinstance(item[key], list) and len(item[key]) >= 4:
                                b = item[key]
                                bboxes.append({
                                    "bbox": [int(b[0]/1000*w), int(b[1]/1000*h), int(b[2]/1000*w), int(b[3]/1000*h)],
                                    "label": str(item.get("label", "field"))
                                })
                                break
    except:
        pass
    
    if not bboxes:
        for match in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text):
            b = [int(x) for x in match]
            bboxes.append({
                "bbox": [int(b[0]/1000*w), int(b[1]/1000*h), int(b[2]/1000*w), int(b[3]/1000*h)],
                "label": "field"
            })
    return bboxes

@GPU_DECORATOR
def predict(image):
    if image is None:
        return None, json.dumps({"error": "画像をアップロードしてください"}, ensure_ascii=False)
    
    try:
        load_model()
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        
        # 画像サイズを制限（メモリ節約）
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        w, h = image.size
        
        messages = [
            {"role": "system", "content": "あなたは日本の書類を分析するエキスパートです。担当者が記入する欄を検出してください。"},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "この画像から担当者が記入する入力フィールドの位置をすべて検出してください。結果はJSON形式で返してください。"}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
        response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        bboxes = parse_output(response, w, h)
        result_img = draw_boxes(image, bboxes)
        
        result_json = {
            "detected": len(bboxes),
            "model": "Qwen3-VL-32B-Instruct + LoRA",
            "fields": [{"id": i+1, "label": b["label"], "bbox": b["bbox"]} for i, b in enumerate(bboxes)],
            "raw": response
        }
        return result_img, json.dumps(result_json, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}\n{traceback.format_exc()}")
        return image if image else None, json.dumps({"error": error_msg}, ensure_ascii=False)

with gr.Blocks() as demo:
    gr.Markdown("# PDFフォームフィールド検出デモ (32B)")
    gr.Markdown("""
    書類画像をアップロードすると、申請者が記入すべきフィールドを検出してbboxを描画します。
    
    **モデル**: Qwen3-VL-32B-Instruct + LoRA (QLoRA fine-tuned)
    """)
    
    with gr.Row():
        input_img = gr.Image(type="pil", label="入力画像")
        output_img = gr.Image(type="pil", label="検出結果")
    
    output_json = gr.Textbox(label="検出結果 (JSON)", lines=10)
    btn = gr.Button("検出実行", variant="primary")
    btn.click(fn=predict, inputs=input_img, outputs=[output_img, output_json])

demo.launch(show_error=True)
