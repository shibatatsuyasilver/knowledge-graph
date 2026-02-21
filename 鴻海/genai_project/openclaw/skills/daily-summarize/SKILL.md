---
name: daily-summarize
description: 讀取 .local/collect.md 列出的檔案並產生摘要，輸出到 .local/summaries.md。
metadata: {"openclaw":{"emoji":"📝","os":["darwin","linux"],"requires":{"bins":["cat","head"]}}}
---

# Daily Summarize

基於 daily-collect 輸出的檔案清單，讀取內容並產生摘要。

## 前置條件

```bash
if [ ! -f .local/collect.md ]; then
  echo "錯誤：請先執行 /daily-collect"
  exit 1
fi
cat .local/collect.md
```

## 執行步驟

### Step 1: 初始化摘要檔案

```bash
echo "# 每日檔案摘要" > .local/summaries.md
echo "" >> .local/summaries.md
echo "**產生時間**: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> .local/summaries.md
echo "" >> .local/summaries.md
```

### Step 2: 逐一讀取並摘要

從 `.local/collect.md` 找出每個路徑後，依序：

1. 用 `head -100` 讀取前 100 行。
2. 判斷副檔名（`.md` / `.csv` / `.json` / 其他）。
3. 將內容交由 LLM 產生 3~5 句繁中摘要。
4. 寫入 `.local/summaries.md`。

範例流程：

```bash
FILE="/path/to/file.md"

echo "## $FILE" >> .local/summaries.md
head -100 "$FILE" >> .local/.tmp_preview.txt
# 由 Agent 讀取 .local/.tmp_preview.txt 並寫入摘要到 summaries.md
```

### Step 3: 驗證輸出

```bash
wc -l .local/summaries.md
cat .local/summaries.md
```

## 完成條件

- `.local/summaries.md` 存在且非空。
- 每個來源檔案都有對應摘要段落。
