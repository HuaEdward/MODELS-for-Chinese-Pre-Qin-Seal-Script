# ZhuanShuInfer — 篆書單字識別（Seal script to Traditional Chinese Characters Inference）

> 繁體中文 / English

## 簡介 (Overview)

**ZhuanShuInfer** 是一個針對古文字／篆書的「單字分類」推理工具。  
你只需準備單字圖片，即可輸出每張圖的 **Top-3 繁體字候選**（含機率）、可視化標註圖片，以及批次結果 CSV。  
本倉庫僅包含**推理與測試**；訓練已在雲端完成（模型權重已提供）。

This repo provides **inference & testing** for a seal-script (Zhuanshu) single-character classifier.  
Given a single-character image, it outputs **Top-3 Traditional Chinese predictions** with probabilities, per-image annotated previews, and a CSV summary. Training is done elsewhere; pre-trained weights are included.

---

## 目前測得準確度 (Current Measured Accuracy)

以 `test_fonts` + 常用字白名單 `common2500.txt` 生成測試集並評估（`gen_testset.py`）：

- **N = 8,699**
- **Top-1 = 73.9395%**
- **Top-3 = 75.4569%**

These numbers come from the provided test font set and the 2,500 common-character whitelist.

---

## 目錄結構 (Folder Structure)

```

ZhuanShuInfer/
├─ data/
│  └─ meta/
│     └─ labels.json          # 類別 ↔ 繁體字 映射（必需 / required）
├─ fonts/                     # 用於可視化標註時顯示中文（選用 / optional）
├─ test\_fonts/                # 測試生成所用的篆書字體（ttf/otf/rar）
├─ predictions/               # 推理輸出（CSV、標註圖片、串聯輸出等）
├─ inscriptions/                   # 你要識別的一批「單字圖片資料夾」就放這裡
│  └─ \<your\_set>/             # e.g. scripts/baoshan\_page01/\*.png
├─ ckpt\_best.pt               # 模型權重（必需 / required）
├─ calibration.json           # 溫度縮放（可無 / optional）
├─ infer.py                   # 單圖/資料夾推理（圖片+CSV，CSV 會「追加寫」）
├─ script\_infer.py            # 一個資料夾整批推理，並輸出 top1 串聯
├─ gen\_testset.py             # 用 test\_fonts + 白名單 生成測試集並評估
├─ common2500.txt             # 2,500 常用字白名單（你提供）
└─ requirements.txt

````

**說明（Important）**
- `ckpt_best.pt`、`data/meta/labels.json` 是推理**必需**檔案。  
- 可視化預覽圖若要正確顯示中文，請放一個 CJK 字體到 `fonts/`（如 `NotoSansCJK-Regular.ttc`）。  
- `scripts/` 是你要識別的任務資料夾集合（原本範例名叫 `rubbings`）。

---

## 安裝需求 (Requirements)

- Python 3.9+
- PyTorch / TorchVision / timm / Pillow / NumPy / OpenCV (optional)
- `rarfile`（若需自動解壓 `.rar` 測試字體；系統需安裝 `unrar/unar`）

安裝（Install）：
```bash
pip install -r requirements.txt
# 若需處理 .rar 測試字體：
# Ubuntu: sudo apt-get install -y unrar
# macOS:  brew install unar
````

---

## 快速開始 (Quick Start)

### 單張圖片推理 (Single Image Inference)

```bash
python infer.py --path scripts/baoshan_page01/001.png --label_font fonts/CJK-Regular.ttf
# 輸出：
#  predictions/annotated/001_pred.png
#  predictions/predictions.csv（追加寫入）
```

### 一個資料夾整批推理 + 串聯 (Batch Folder + Top-1 Concatenation)

把要識別的單字圖放在 `inscriptions/<set_name>/` 下：

```bash
python inscription_infer.py --dir inscriptions/baoshan_page01 --label_font fonts/CJK-Regular.ttf
# 輸出：
#  predictions/baoshan_page01/annotated/*.png
#  predictions/baoshan_page01/predictions.csv
#  predictions/baoshan_page01/top1_concat.txt   # 所有字的 Top-1 串起來
#  predictions/baoshan_page01/top1_concat.png   # 串聯行的預覽圖
```

> 檔名會按「自然排序」串聯（e.g. 1, 2, 10），若要精確的原文順序，請在檔名中自行編號。

### 只追加 CSV 不覆蓋 (Append CSV)

`infer.py` 已改為**追加寫**；若要重置歷史：

```bash
rm -f predictions/predictions.csv
```

---

## 生成測試集並評估 (Generate Test Set & Evaluate)

利用 `test_fonts/` 的篆書字體 + 你提供的 **2,500 常用字白名單** `common2500.txt`：
（白名單可為簡/繁，預設會做「簡→繁」統一；如已是繁體可加 `--no_s2t`）

```bash
# 建議先抽樣 300 字快速測
python gen_testset.py \
  --fonts_dir test_fonts \
  --whitelist common2500.txt \
  --max_chars 300 \
  --fonts_per_char 2 \
  --samples_per_font 1 \
  --arch resnet50

# 全量 2,500 字（可能較久）
python gen_testset.py \
  --fonts_dir test_fonts \
  --whitelist common2500.txt \
  --fonts_per_char 2 \
  --samples_per_font 1 \
  --arch resnet50
```

輸出（Outputs）：

* 生成的測試圖：`data/test/*.png`（檔名形如 `<labelid>_<font>_<k>.png`）
* 自動評估 Top-1 / Top-3 並列印於終端；統計資訊寫入 `data/test/meta_test.json`

---

## 常見問題 (FAQ)

**Q1. 標註圖片是空白？**
`infer.py / script_infer.py` 已內建自動對比與二值化，並有反相兜底；若仍偏白，請確認輸入為「單字」且背景對比足夠。

**Q2. 可視化中文字顯示成方塊？**
請在 `--label_font` 指定一個 CJK 字體，或把字體放到 `fonts/` 目錄（腳本會自動尋找）。

**Q3. 機率不可信（過度自信）？**
若有 `calibration.json` 會自動進行溫度縮放（T）；沒有也不影響 Top-k 排名，只是數值可能偏保守/自信。

**Q4. 主幹網路（architecture）需一致嗎？**
是。若訓練時不是 `resnet50`，推理與評估時請加 `--arch <your_arch>`。

---

## 授權 (License)

僅供研究與個人使用。字體檔之授權以各自原始授權為準。
For research & personal use. Font files comply with their respective licenses.

---