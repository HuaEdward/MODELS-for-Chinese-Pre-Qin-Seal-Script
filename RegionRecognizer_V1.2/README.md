# Pre-Qin Seal-Script Region Classifier (Training+Inference) — 先秦篆書國別分類器（訓練＋推理）

目前訓練數據集並不完備（特別是燕國文字），本人只是高中生，希望更多同道能提供一些幫助！
The current training dataset is incomplete (especially the Yan script). I am only a high school student and hope that more ancient script researchers can provide some help!

> 繁體中文 / English


## 簡介 (Overview)

此專案用於判斷**先秦篆書**（單字或整段碑文）更可能屬於 **秦 / 齊 / 燕 / 楚 / 三晉** 之哪一方國別。  
- **單字層級**：輸入單字影像，輸出五國機率與可視化圖。:contentReference[oaicite:0]{index=0}  
- **碑文層級**：輸入一個資料夾（含多張單字圖），對其**整體**做國別歸屬推斷，並輸出總表與圖表。:contentReference[oaicite:1]{index=1}  
- 亦提供資料前處理腳本，將原圖規一化為**黑底白字、等比置中**之訓練輸入。:contentReference[oaicite:2]{index=2}

This project classifies **Pre-Qin seal-script** (single characters or whole inscriptions) into one of **Qin / Qi / Yan / Chu / SanJin**.  
- **Single character**: image → probabilities over 5 states + visualization. :contentReference[oaicite:3]{index=3}  
- **Whole inscription**: a folder of characters → overall attribution + CSV and plots. :contentReference[oaicite:4]{index=4}  
- Includes preprocessing to normalize images to **white on black, centered, fixed size**. :contentReference[oaicite:5]{index=5}

---

## 專案結構 (Project Structure)

```

.
├── data/                          # 訓練資料（按國別分子資料夾）
│   ├── Qin/
│   │   ├── input\_images/          # 原始圖
│   │   └── output\_images/         # 前處理後圖（用於訓練）
│   ├── Qi/  ...（Yan / Chu / SanJin 同結構）
│
├── single\_character\_prediction/    # 單字批次推理輸入/輸出
│   ├── input/
│   └── output/
├── inscription\_prediction/         # 碑文（整段）推理輸入/輸出
│   ├── input/
│   └── output/
│
├── data\_prep.py                    # 前處理：灰階→二值→白字黑底→等比置中 128×128 &#x20;
├── train\_model.py                  # 訓練腳本（定義 AncientTextClassifier）
├── single\_character\_predict.py     # 單字推理（繪製機率長條圖、輸出 CSV）    &#x20;
├── inscription\_predict.py          # 碑文推理（整體機率＝多字平均+正規化）    &#x20;
├── ancient\_text\_classifier.pth     # 訓練後模型（或 best\_model.pth）
├── model\_info.json                 # 模型與訓練配置
├── training\_history.png            # 訓練過程曲線
└── requirements.txt

````

---

## 安裝需求 (Requirements)

- Python 3.9+
- 主要套件：`opencv-python`, `numpy`, `pandas`, `matplotlib`, `torch`, `torchvision`
- 建議使用 GPU 的 PyTorch 版本以加速推理

**English:**  
- Python 3.9+  
- Core packages: `opencv-python`, `numpy`, `pandas`, `matplotlib`, `torch`, `torchvision`  
- GPU-enabled PyTorch is recommended for faster inference

Install:
```bash
pip install -r requirements.txt
````

---

## 資料前處理 (Data Preprocessing)

**目的**：將任意底色/大小的字形影像，轉為**白字黑底**、維持比例、置中到 **128×128**，並輸出到各國別的 `output_images/`。
**用法**：

```bash
python data_prep.py
```

* 程式會遍歷 `data/<State>/input_images/`，輸出到 `data/<State>/output_images/`。
* 前處理包含：灰階讀取 → 二值化 → 必要時反相（統一白字黑底） → 等比縮放與置中。

**English:**

* **Goal:** Normalize any input glyph to **white-on-black**, aspect-preserved, centered **128×128** canvases, and write to each state’s `output_images/`.
* **Usage:** `python data_prep.py`
* The script scans `data/<State>/input_images/` and saves to `data/<State>/output_images/`.
* Steps: grayscale → binarize → invert if needed (unify white-on-black) → scale with aspect ratio and center.

This script walks through `data/<State>/input_images/` and writes normalized images to `.../output_images/`, standardizing to 128×128 white-on-black, centered canvases.&#x20;

---

## 訓練 (Training)

```bash
python train_model.py
```

輸出將包含：

* `ancient_text_classifier.pth`（或 `best_model.pth`）
* `model_info.json`
* `training_history.png`

> 若你已提供訓練完成的 `*.pth`，可直接進入「推理」步驟。

**English:**
Running `python train_model.py` produces:

* `ancient_text_classifier.pth` (or `best_model.pth`)
* `model_info.json`
* `training_history.png`

> If you already have a trained `*.pth`, you can skip training and go straight to inference.

---

## 單字推理 (Single-Character Inference)

### 互動模式（CLI）

```bash
python single_character_predict.py
```

功能選單包含：

1. 單張圖片預測 → 顯示五國機率、生成對應圖表與處理後影像
2. 批次預測 `single_character_prediction/input/` → 輸出到 `.../output/`
3. 檢視 I/O 資料夾狀態
   （流程與輸出檔名：見程式內 `predict_batch()` 與 `create_result_visualization()`，含圖表與 CSV 匯出。）

**English:**
Interactive menu:

1. Predict a single image → show probabilities for all regions and save chart + processed glyph
2. Batch predict images under `single_character_prediction/input/` → outputs to `.../output/`
3. Inspect I/O folders (details and filenames implemented in `predict_batch()` and `create_result_visualization()`; charts and CSV are exported)

**批次推理直接用法**：

```bash
python -c "from single_character_predict import Predictor; Predictor().predict_batch('single_character_prediction/input','single_character_prediction/output')"
```

* 可視化圖：`*_result.png`
* 前處理影像：`*_processed.png`
* 明細 CSV：`prediction_results_YYYYMMDD_HHMMSS.csv`（含各國機率與 Top1）

**English:**

* Visualization: `*_result.png`
* Preprocessed glyph: `*_processed.png`
* Detailed CSV: `prediction_results_YYYYMMDD_HHMMSS.csv` (includes per-region probabilities and Top-1)

The `Predictor` loads `ancient_text_classifier.pth` or `best_model.pth`, runs the same preprocessing as training (white-on-black 128×128), and exports charts + CSV.&#x20;

---

## 碑文（整段）推理 (Whole-Inscription Inference)

將同一段碑文的**所有單字圖**放在一個子資料夾，例如：

```
inscription_prediction/
└─ input/
   ├─ stele_A/
   │  ├─ 001.png
   │  ├─ 002.png
   │  └─ ...
   └─ stele_B/
      └─ ...
```

執行：

```bash
python inscription_predict.py
```

每個碑文資料夾會得到：

* **整體機率圖**：`input_overall_prediction.png`
* **逐字明細表**：`input_detailed_predictions.csv`（每張單字圖的五國機率）
* 請見 `InscriptionPredictor.predict_inscription()`：對所有單字的機率取平均，再正規化為百分比作為整段預測。

**English:**
Place all single-character crops of one inscription in a subfolder under `inscription_prediction/input/`, then run `python inscription_predict.py`.
For each inscription you’ll get:

* `input_overall_prediction.png` — overall regional probabilities
* `input_detailed_predictions.csv` — per-character probabilities
* Implementation detail: `InscriptionPredictor.predict_inscription()` averages per-character probabilities and normalizes to percentages.

---

## I/O 與輸出格式 (I/O & Outputs)

* **支援影像格式**：PNG / JPG / JPEG / BMP / TIFF（單字圖即可）
* **單字輸出**（single\_character\_prediction/output）：

  * `*_result.png`：左側顯示預處理影像，右側為五國機率長條圖
  * `*_processed.png`：128×128 規一化影像
  * `prediction_results_*.csv`：每張圖的 Top1 與各國機率（%）
* **碑文輸出**（inscription\_prediction/output/<inscription>/）：

  * `input_overall_prediction.png`：整段平均後的國別機率（%）
  * `input_detailed_predictions.csv`：逐字機率明細表（同資料夾內全部單字）

**English:**

* **Supported formats:** PNG / JPG / JPEG / BMP / TIFF
* **Single-character outputs** (`single_character_prediction/output`):

  * `*_result.png` (processed glyph on the left; bar chart of regional probabilities on the right)
  * `*_processed.png` (normalized 128×128 glyph)
  * `prediction_results_*.csv` (Top-1 and all region probabilities)
* **Inscription outputs** (`inscription_prediction/output/<inscription>/`):

  * `input_overall_prediction.png` (overall regional percentages)
  * `input_detailed_predictions.csv` (per-glyph probability table)

---

## 常見問題 (FAQ)

**Q1. 圖片要怎麼準備？**
盡量裁到**單字**，背景簡潔。若不確定，可先放到 `data/<State>/input_images/`，跑一次 `data_prep.py` 看規一化效果。

**English:**
**Q1. How should I prepare images?**
Crop to **single characters** with clean backgrounds. If unsure, place them in `data/<State>/input_images/` and run `data_prep.py` to preview normalization.

**Q2. 推理結果為何是百分比且總和近 100%？**
分類輸出經 softmax 轉為百分比。單字與碑文模式皆輸出五國百分比；碑文模式對多張單字取平均後再正規化。

**English:**
**Q2. Why percentages summing to \~100%?**
Outputs are softmax probabilities. Both single-character and inscription modes report regional percentages; inscription mode averages per-glyph results then normalizes.

**Q3. 批次推理與互動式有何差別？**
互動式會詢問操作並顯示即時結果；批次模式直接掃描指定資料夾並匯出圖表與 CSV，便於大量處理。

**English:**
**Q3. Difference between batch and interactive?**
Interactive prompts for actions and displays immediate results; batch scans a folder and exports charts + CSV for large-scale processing.

**Q4. 模型檔案找不到？**
確保根目錄下存在 `ancient_text_classifier.pth` 或 `best_model.pth`；否則先執行 `train_model.py` 訓練。

**English:**
**Q4. Model file not found?**
Ensure `ancient_text_classifier.pth` or `best_model.pth` exists in the project root, or run `train_model.py` first.

---

## 引用與備註 (Notes)

* 本專案之程式流程與 I/O 介面以三支核心腳本為準：
  `data_prep.py`（前處理）、`single_character_predict.py`（單字推理）、`inscription_predict.py`（碑文推理）。
  內含之細節（例如規一化方式、輸出檔名、平均策略）可於程式碼中查閱。

**English:**
This project’s pipeline and I/O are defined by three core scripts:
`data_prep.py` (preprocessing), `single_character_predict.py` (single-character inference), and `inscription_predict.py` (inscription inference).
See code for details such as normalization, filenames, and averaging strategies.

---

## 授權與致謝 (License & Acknowledgements)

* 僅供研究與個人使用。資料與字體之授權以其原始授權為準。

* 若你願意分享資料或改進模型，非常感謝你的幫忙！（特別是補充燕國篆書字例 🙏）

**English:**

* For research and personal use only. Dataset/font licenses follow their original terms.
* Contributions of data and model improvements are warmly welcomed—especially additional **Yan** exemplars 🙏
