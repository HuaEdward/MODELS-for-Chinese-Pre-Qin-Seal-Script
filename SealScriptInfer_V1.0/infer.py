import os, sys, glob, csv, json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch, torch.nn as nn
import torchvision.transforms as T
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

                                         

def list_images(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        files = []
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.webp"):
            files += glob.glob(str(p / ext))
        return sorted(files)
    if p.is_file():
        return [str(p)]
          
    files = glob.glob(input_path)
    return sorted(files)

def safe_load_state_dict(model, ckpt_path="ckpt_best.pt"):
                               
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)

def load_label_map():
    with open("data/meta/labels.json","r",encoding="utf-8") as f:
        label2id = json.load(f)
                                         
    id2label = {int(v): k for k,v in label2id.items()}
    return id2label

def find_font(path_hint=None):
               
    if path_hint and Path(path_hint).exists():
        try:
            return ImageFont.truetype(path_hint, size=28)
        except Exception:
            pass
                        
    for ext in ("*.ttf","*.otf","*.ttc"):
        for fp in Path("fonts").rglob(ext):
            try:
                return ImageFont.truetype(str(fp), size=28)
            except Exception:
                continue
    return ImageFont.load_default()

                                         

class Net(nn.Module):
    def __init__(self, num_classes, arch="resnet50"):
        super().__init__()
        self.m = timm.create_model(arch, pretrained=False, in_chans=1, num_classes=num_classes)
    def forward(self, x): return self.m(x)

def load_model(arch="resnet50"):
    id2label = load_label_map()
    model = Net(len(id2label), arch=arch).to(DEVICE)
    safe_load_state_dict(model, "ckpt_best.pt")
    model.eval()
    Tval = 1.0
    cal = Path("calibration.json")
    if cal.exists():
        try:
            with open(cal,"r") as f:
                Tval = float(json.load(f).get("T", 1.0))
        except Exception:
            Tval = 1.0
    return model, id2label, Tval

                                                 

def auto_contrast(arr):
                            
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1:          
        return arr
    out = (arr - p1) * 255.0 / (p99 - p1)
    return np.clip(out, 0, 255).astype(np.uint8)

def binarize(arr):
                               
    try:
        import cv2
        _, thr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = thr
    except Exception:
        t = 0.9 * float(arr.mean()) + 0.1 * float(arr.min())
        bin_img = (arr > t).astype(np.uint8) * 255
                         
    black_ratio = (bin_img == 0).mean()
    if black_ratio < 0.002 or black_ratio > 0.98:
              
        bin_img = 255 - bin_img
    return bin_img

def preprocess(png_path):
                                     
    img = Image.open(png_path).convert("L")
    arr = np.array(img)
    arr = auto_contrast(arr)
    bin_img = binarize(arr)

    pil = Image.fromarray(bin_img)
                                          
    bbox = ImageOps.invert(pil).getbbox()
    if bbox:
        pil = pil.crop(bbox)
    pil = ImageOps.expand(pil, border=12, fill=255).resize((224,224), Image.BILINEAR)

                                
    if np.array(pil).mean() > 250:
        pil = Image.fromarray(255 - bin_img)
        bbox = ImageOps.invert(pil).getbbox()
        if bbox:
            pil = pil.crop(bbox)
        pil = ImageOps.expand(pil, border=12, fill=255).resize((224,224), Image.BILINEAR)

    tensor = T.Compose([T.ToTensor(), T.Normalize([0.5],[0.5])])(pil).unsqueeze(0)
    return pil, tensor

                                                   

def topk_predict(model, x, k, Tval):
    with torch.no_grad():
        logits = model(x.to(DEVICE)) / Tval
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = prob.argsort()[-k:][::-1]
    return [(int(i), float(prob[int(i)])) for i in idx]

def annotate(img224, pairs, font):
                                          
    H = 224
    W_left = 224
    W_right = 380
    canvas = Image.new("RGB", (W_left + W_right, H), (255,255,255))
    canvas.paste(img224.convert("RGB"), (0,0))

    panel = Image.new("RGB", (W_right, H), (16,16,16))
    canvas.paste(panel, (W_left, 0))
    draw = ImageDraw.Draw(canvas)
    x0, y0 = W_left + 16, 16
    draw.text((x0, y0), "Top-3", fill=(240,240,240), font=font)
    y = y0 + 36
    for idx,(ch,p) in enumerate(pairs, start=1):
        line = f"{idx}) {ch}    p={p:.4f}"
        draw.text((x0, y), line, fill=(255,255,255), font=font)
        y += 36
    return canvas

                                        

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="单图 / 文件夹 / 通配符")
    ap.add_argument("--out", default="predictions", help="输出目录")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arch", type=str, default="resnet50")
    ap.add_argument("--label_font", type=str, default=None, help="用于在图片上显示中文的字体文件")
    args = ap.parse_args()

    files = list_images(args.path)
    if not files:
        print("No images found:", args.path); sys.exit(1)

    out_dir = Path(args.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    model, id2label, Tval = load_model(arch=args.arch)
    font = find_font(args.label_font)
    
    csv_path = out_dir / "predictions.csv"
    header = ["image","top1_char","top1_prob","top2_char","top2_prob","top3_char","top3_prob"]

                             
    need_header = not (csv_path.exists() and csv_path.stat().st_size > 0)

    with open(csv_path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if need_header:
            writer.writerow(header)
        for f in files:
            try:
                img224, x = preprocess(f)
                preds = topk_predict(model, x, args.k, Tval)
                       
                pairs = []
                for i,p in preds:
                    ch = id2label.get(i, "")
                    pairs.append((ch, p))

                     
                vis = annotate(img224, pairs, font)
                vis.save(out_dir / "annotated" / (Path(f).stem + "_pred.png"))

                                
                def g(n):
                    return pairs[n] if n < len(pairs) else ("", 0.0)
                (c1,p1) = g(0); (c2,p2) = g(1); (c3,p3) = g(2)
                writer.writerow([f, c1, f"{p1:.6f}", c2, f"{p2:.6f}", c3, f"{p3:.6f}"])

            except Exception as e:
                                
                writer.writerow([f, "ERROR", str(e), "", "", "", ""])
                print("[WARN]", f, "->", e)

    print(f"[DONE] 图像输出：{out_dir/'annotated'}")
    print(f"[DONE] CSV：{csv_path}")

if __name__ == "__main__":
    main()
