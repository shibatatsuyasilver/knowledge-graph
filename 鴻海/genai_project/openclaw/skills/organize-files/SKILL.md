---
name: organize-files
description: 將指定目錄中的雜亂檔案依副檔名歸檔至對應資料夾 (PDF -> docs, JPG -> images)。
metadata:
  openclaw:
    emoji: "🗂️"
    os: ["darwin", "linux"]
    requires:
      bins: ["mv", "mkdir", "find"]
---

# Organize Files Skill

整理 `~/Downloads` 或指定目錄的檔案。

## 功能

將 `.pdf`, `.docx` 移動至 `documents/`
將 `.jpg`, `.png` 移動至 `images/`

## 執行邏輯

```bash
TARGET_DIR="${1:-$HOME/Downloads}"
DOC_DIR="$TARGET_DIR/documents"
IMG_DIR="$TARGET_DIR/images"

# 建立目標資料夾
mkdir -p "$DOC_DIR" "$IMG_DIR"

# 移動文件
count_docs=0
find "$TARGET_DIR" -maxdepth 1 -name "*.pdf" -o -name "*.docx" | while read f; do
    mv "$f" "$DOC_DIR/"
    ((count_docs++))
done

# 移動圖片
count_imgs=0
find "$TARGET_DIR" -maxdepth 1 -name "*.jpg" -o -name "*.png" | while read f; do
    mv "$f" "$IMG_DIR/"
    ((count_imgs++))
done

# 輸出結果供後續技能使用
echo "Organized Files Report" > /tmp/organize_report.txt
echo "----------------------" >> /tmp/organize_report.txt
echo "Documents moved: $count_docs" >> /tmp/organize_report.txt
echo "Images moved:    $count_imgs" >> /tmp/organize_report.txt
echo "Timestamp:       $(date)" >> /tmp/organize_report.txt
```
