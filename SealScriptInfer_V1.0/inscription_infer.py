import os, sys, csv, json, glob, re
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch, torch.nn as nn
import torchvision.transforms as T
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

                                         
def natural_key(s: str):
                               
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_dir_images(dir_path: str):
    p = Path(dir_path)
    if not p.is_dir():
        raise FileNotFoundError(f"{dir_path} 不是有效的目录")
    files = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.webp"):
        files += glob.glob(str(p / ext))
    files = sorted(files, key=natural_key)
    return files

def safe_load_state_dict(model, ckpt_path="ckpt_best.pt"):
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)

def load_label_map():
    with open("data/meta/labels.json","r",encoding="utf-8") as f:
        label2id = json.load(f)
    return {int(v): k for k, v in label2id.items()}

def find_font(path_hint=None, size=36):
    if path_hint and Path(path_hint).exists():
        try:
            return ImageFont.truetype(path_hint, size=size)
        except Exception:
            pass
    for ext in ("*.ttf","*.otf","*.ttc"):
        for fp in Path("fonts").rglob(ext):
            try:
                return ImageFont.truetype(str(fp), size=size)
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
    if bbox: pil = pil.crop(bbox)
    pil = ImageOps.expand(pil, border=12, fill=255).resize((224,224), Image.BILINEAR)
    if np.array(pil).mean() > 250:
        pil = Image.fromarray(255 - bin_img)
        bbox = ImageOps.invert(pil).getbbox()
        if bbox: pil = pil.crop(bbox)
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
    H = 224; W_left = 224; W_right = 420
    canvas = Image.new("RGB", (W_left + W_right, H), (255,255,255))
    canvas.paste(img224.convert("RGB"), (0,0))
    panel = Image.new("RGB", (W_right, H), (16,16,16))
    canvas.paste(panel, (W_left, 0))
    draw = ImageDraw.Draw(canvas)
    x0, y0 = W_left + 16, 16
    draw.text((x0, y0), "Top-3", fill=(240,240,240), font=font)
    y = y0 + 40
    for idx,(ch,p) in enumerate(pairs, start=1):
        line = f"{idx}) {ch}    p={p:.4f}"
        draw.text((x0, y), line, fill=(255,255,255), font=font)
        y += 36
    return canvas

def render_concat_image(chars, font, out_png):
                     
    text = "".join(chars)
    if not text:
        return
            
    char_w = font.getlength("漢") if hasattr(font, "getlength") else font.getsize("漢")[0]
    W = int(max(320, char_w * max(8, len(chars)) * 0.9))
    H = 80
    img = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(0,0,0), font=font)
    img.save(out_png)

                                        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="包含一整块拓片的所有单字图的文件夹")
    ap.add_argument("--out", default=None, help="输出目录（默认：predictions/<dirname>）")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arch", type=str, default="resnet50")
    ap.add_argument("--label_font", type=str, default=None, help="用于中文标注与串联图的字体文件")
    ap.add_argument("--append", action="store_true", help="predictions.csv 若存在则追加（默认覆盖写）")
    ap.add_argument("--master_csv", type=str, default=None, help="可选：把结果再追加写入这个“总表”CSV")
    args = ap.parse_args()

    files = list_dir_images(args.dir)
    if not files:
        print("目录里没有找到图片：", args.dir); sys.exit(1)

                                  
    if args.out is None:
        out_dir = Path("predictions") / Path(args.dir).name
    else:
        out_dir = Path(args.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "predictions.csv"
    header = ["image","top1_char","top1_prob","top2_char","top2_prob","top3_char","top3_prob"]

    model, id2label, Tval = load_model(arch=args.arch)
    font = find_font(args.label_font, size=32)

              
    write_mode = "a" if (args.append and csv_path.exists()) else "w"
    need_header = (write_mode == "w")

             
    with open(csv_path, write_mode, newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if need_header:
            writer.writerow(header)

                            
        master_writer = None
        master_fp = None
        if args.master_csv:
            master_path = Path(args.master_csv)
            master_path.parent.mkdir(parents=True, exist_ok=True)
            m_need_header = not (master_path.exists() and master_path.stat().st_size > 0)
            master_fp = open(master_path, "a", newline="", encoding="utf-8")
            master_writer = csv.writer(master_fp)
            if m_need_header:
                master_writer.writerow(["folder"] + header)

        concat_top1 = []

        for f in files:
            try:
                img224, x = preprocess(f)
                preds = topk_predict(model, x, args.k, Tval)
                pairs = [(id2label.get(i, ""), p) for (i,p) in preds]

                     
                vis = annotate(img224, pairs, font)
                vis.save(out_dir / "annotated" / (Path(f).stem + "_pred.png"))

                     
                def g(n):
                    return pairs[n] if n < len(pairs) else ("", 0.0)
                (c1,p1) = g(0); (c2,p2) = g(1); (c3,p3) = g(2)
                row = [f, c1, f"{p1:.6f}", c2, f"{p2:.6f}", c3, f"{p3:.6f}"]
                writer.writerow(row)
                if master_writer:
                    master_writer.writerow([args.dir] + row)

                         
                concat_top1.append(c1)

            except Exception as e:
                               
                writer.writerow([f, "ERROR", str(e), "", "", "", ""])
                if master_writer:
                    master_writer.writerow([args.dir, f, "ERROR", str(e), "", "", "", ""])
                print("[WARN]", f, "->", e)

                
        concat_text = "".join(concat_top1)
        (out_dir / "top1_concat.txt").write_text(concat_text, encoding="utf-8")
                    
        render_concat_image(concat_top1, font, out_dir / "top1_concat.png")

        if master_fp:
            master_fp.close()

    print(f"[DONE] 标注图：{out_dir/'annotated'}")
    print(f"[DONE] CSV：{csv_path}")
    print(f"[DONE] 串联文本：{out_dir/'top1_concat.txt'}")
    print(f"[DONE] 串联预览图：{out_dir/'top1_concat.png'}")
    print(f"[INFO] 串联Top-1：{concat_text}")
                    
    print(concat_text)

if __name__ == "__main__":
    main()
