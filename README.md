# DATASETS-and-MODELS-for-Chinese-Pre-Qin-Script
This project includes: data collection for Pre-Qin scripts; a model for inferring the origin of the Chinese nation(Qin, Chu, Yan, and the Three Jins); a Pre-Qin seal script recognition and translation model; and a generative model for converting modern characters into the Pre-Qin style. We welcome contributions from more ancient script researchers!

本項目包括：先秦文字數據收集；先秦文字所屬國家（秦、楚、燕、三晉）起源推斷模型；先秦篆書識別與翻譯模型；以及現代文字轉為先秦文字的生成模型。歡迎更多古文字研究者貢獻力量！

---

## Overview · 專案總覽

- **(A) Pre-Qin Seal Script Recognition → Traditional**  
  **先秦篆書識別與翻譯（輸出繁體）**  
  Input a cropped Pre-Qin/Seal character image and get **Top-k Traditional candidates with probabilities**. Built for research & annotation assistance.

- **(B) Regional Origin Classifier (Qin / Chu / Yan / San-Jin)**  
  **先秦文字所屬國起源分類器（秦、楚、燕、三晉）**  
  Classifies an inscription/rubbing image into one of the main cultural regions to aid provenance studies.

- **(C) Modern-to-Pre-Qin Generator (Experimental)**  
  **現代繁體 → 先秦風格生成（試驗中）**  
  Currently an **attempt on Bao-Shan Chu slips** only; quality is not yet stable. Needs more paired data and fonts to improve.

> 📌 Data status note · 數據狀態說明  
> 目前訓練數據集並不完備（**特別是燕國文字**）。我目前仍在高中求學階段，能力有限，**真誠期待各位同道的參與與指正**。
> The current training dataset is incomplete (especially the Yan script). I am still in high school and my skills are limited. I sincerely look forward to your participation and corrections.

---

## Why this repository? · 我們在做什麼

- **Unify scattered materials** of Pre-Qin scripts into machine-learnable datasets.  
- **Build open baselines** for recognition, translation (to Traditional), and regional classification.  
- **Explore generation** from modern Traditional to historical styles to assist reading and pedagogy.  

- **整合分散材料**成可機器學習的先秦文字數據集。  
- **建立開源基線模型**：識別、翻譯（輸出繁體）、地域分類。  
- **探索生成模型**：由現代繁體推測還原先秦風貌，用於教學與研究輔助。

---

## Current Status · 目前進度

- (A) **Seal→Traditional Recognizer**：可用，仍需擴充字表與樣本來源；支援 Top-k 候選輸出。
- Available, but still needs to expand the vocabulary and sample sources， in order to support Top-k candidate output.
- (B) **Region Classifier**：可用，但是數據集資料稀缺，需更多文字樣本增加泛化。
- Available, but the dataset is scarce and requires more script samples to improve generalization.
- (C) **Modern→Pre-Qin Generator**：僅在**包山楚簡**上做了初步實驗，效果**尚不穩定**。已規劃採用**合成預訓練＋小樣本微調**、配對增廣等方法持續改進。暫時未開源。
- Initial experiments have been conducted on the Baoshan Chu Bamboo Slips, and the results are not yet stable. I plan to continue improving this model using methods such as synthetic pre-training + small sample fine-tuning and paired augmentation. This model is not yet open source.

---

## Contributing Data · 數據貢獻指南

We warmly welcome dataset contributions. Below are **minimal formats** that our tools already support:

我們誠摯邀請貢獻數據。以下為工具已支援的**基礎格式**：

### (1) Character-to-Traditional Pairs ·「字形→繁體」對齊
- **Folder-per-char**: a directory named by the **Traditional character** (e.g., `國/`), containing images of that character in Pre-Qin/Seal style.  
- Include various sources: **rubbings, inscriptions, bamboo/slip photos, tracings**.  
- Optional CSV per folder: `source, era, material, note`.

- **按字建目錄**：以**繁體字**命名資料夾（如 `國/`），內含該字之先秦/篆書圖像。  
- 接受多種載體：**拓片、金文、竹簡帛書摹本、照片**等。  
- 可選提供 CSV：`source, era, material, note`。

### (2) Regional Labels · 地域標註
- Organize images into top-level folders: `Qin/`, `Chu/`, `Yan/`, `SanJin/`.  
- If possible, add subfolders by site/collection and a manifest CSV.

- 按國別置於 `Qin/`, `Chu/`, `Yan/`, `SanJin/`。  
- 鼓勵按遺址/館藏分子目錄，並附清單 CSV。

### (3) Font Assets · 字體資源
- Provide TTF/OTF for **Traditional** and any **Seal/clerical/bronze-style** fonts with license notes.  
- Mixed mappings (Simplified→Seal) are fine; we handle code-point fallbacks.

- 歡迎提供 **繁體**與**篆/隸/金文風** TTF/OTF（註明授權）。  
- 若為簡→篆的映射也可，我們已支援**繁→簡回退**。

> 🔐 Licensing · 授權  
> 若無特別標註，本倉庫建議：**code = MIT**；**datasets = CC BY-NC 4.0**。若有特殊條款，請在資料夾內附上 LICENSE/README。
> Unless otherwise noted, this repository recommends: **code = MIT**; **datasets = CC BY-NC 4.0**. If there are special terms, please include a LICENSE/README file in the folder.

---

## How to Use · 快速使用

Each subproject includes its own README with environment and commands. Typical stack:

各子項均附使用說明。常見環境：

- **Python 3.10+**, **PyTorch + CUDA**, `torchvision`, `Pillow`, `fontTools`, `OpenCV(headless)`, `scikit-learn`.  
- Clone and follow each module’s README to **prepare data → train → infer/evaluate**.

---

## Roadmap · 後續計畫

- **Data expansion**: enlarge coverage for **Yan** and under-represented graphs/variants.  
- **Better pairing**: curated Traditional↔Seal pairs with provenance metadata.  
- **Generator improvements**: synthetic pretraining, stronger paired augmentation, feature-matching losses...  
- **Evaluation**: establish public test splits & metrics.  

- **數據擴充**：特別是**燕國**與稀見字形。  
- **精確對齊**：繁體↔篆書嚴格配對並附出處。  
- **生成模型**：合成預訓練、配對增廣、特徵匹配...
- **評測體系**：公開測試集與指標。  

---

## Ethics & Disclaimer · 學術倫理與免責

- Images and fonts **must respect copyrights and collection policies**. Provide source and license where possible.  
- Model outputs for ancient scripts **may be uncertain**; always verify with experts.  
- The project is a community effort; **errors and gaps are expected**. Issues and PRs are welcome.

- 圖像/字體須遵守**版權與館藏條款**，盡量附來源與授權。  
- 古文字自動化結果**存在不確定性**，請以專家審校為準。  
- 本倉庫為社群協作，**難免錯漏**，歡迎提交 Issue/PR 共同改進。

---

## Get Involved · 參與方式

- Open a **GitHub Issue/Discussion** to propose datasets, corrections, or features.  
- Submit a **Pull Request** with new data (see formats above) or code improvements.  
- Share references & catalogs helpful for pairing Traditional↔Seal variants.

- 透過 **Issue/Discussion** 提出資料、校勘或功能建議。  
- 提交 **PR** 追加數據或改進代碼（依格式）。  
- 分享有助於繁↔篆對勘的目錄、索引與文獻。

---

## Acknowledgements · 致謝

Thanks to all contributors, collectors, and scholars whose work enables computational study of ancient Chinese scripts.  
感謝所有提供素材與指導的研究者與館藏單位，讓先秦古文字的計算研究成為可能。

