---
name: file-digest
description: 讀取指定檔案或目錄內容並產生結構化摘要。
metadata: {"openclaw":{"emoji":"📄","os":["darwin","linux"],"requires":{"bins":["cat"]}}}
---

# File Digest Skill

讀取檔案內容並產生摘要。

## 觸發條件

- 幫我看一下這個檔案
- 摘要 /path/to/file.md
- 這份報告在說什麼？
- 整理這個目錄的檔案

## 讀取單一檔案

```bash
# 讀取前 200 行避免過長
head -200 "/path/to/file"

# 檢查檔案大小與類型
file "/path/to/file"
wc -l "/path/to/file"
```

## 目錄總覽

```bash
# 列出目錄結構（限兩層）
find "/path/to/directory" -maxdepth 2 -type f | head -30

# 各檔案大小摘要
ls -lhS "/path/to/directory" | head -20
```

## CSV 快速分析

```bash
# 欄位名稱（第一行）
head -1 "/path/to/file.csv"

# 資料筆數
wc -l "/path/to/file.csv"

# 前 5 筆資料
head -6 "/path/to/file.csv"
```

## JSON 快速分析

```bash
python3 -m json.tool "/path/to/file.json" | head -80
```

## 安全限制

- 僅讀取，不修改任何檔案。
- 避免讀取敏感檔案（`.env`, `credentials.json`, `id_rsa`）。
- 檔案超過 500 行時僅讀取前 200 行。
- 先使用 `file` 確認是否為文字檔，避免二進位內容。
