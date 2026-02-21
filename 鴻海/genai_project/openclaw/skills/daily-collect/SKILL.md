---
name: daily-collect
description: 蒐集每日待處理檔案與系統狀態，輸出到 .local/collect.md。
metadata: {"openclaw":{"emoji":"📥","os":["darwin","linux"],"requires":{"bins":["find","date"]}}}
---

# Daily Collect

蒐集指定目錄中過去 24 小時內修改的檔案，並記錄系統狀態。

## 安全規則

- 不修改任何原始檔案。
- 僅讀取指定工作目錄。
- 所有輸出寫入 `.local/`。

## 執行步驟

### Step 1: 建立輸出目錄

```bash
mkdir -p .local
```

### Step 2: 記錄時間戳記

```bash
echo "# 每日資料蒐集報告" > .local/collect.md
echo "" >> .local/collect.md
echo "**蒐集時間**: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> .local/collect.md
echo "**系統 Uptime**: $(uptime)" >> .local/collect.md
echo "" >> .local/collect.md
```

### Step 3: 蒐集今日修改檔案

```bash
TARGET_DIR="${TARGET_DIR:-$HOME/Documents/reports}"

echo "## 今日修改的檔案" >> .local/collect.md
echo "" >> .local/collect.md

find "$TARGET_DIR" -type f -mtime -1 \( -name "*.md" -o -name "*.csv" -o -name "*.json" \) | while read -r f; do
  echo "- \`$f\` ($(wc -c < "$f") bytes)" >> .local/collect.md
done

echo "" >> .local/collect.md
```

### Step 4: 記錄磁碟狀態

```bash
echo "## 磁碟狀態" >> .local/collect.md
df -h / | tail -1 | awk '{print "- 使用: "$3" / 總計: "$2" / 可用: "$4}' >> .local/collect.md
echo "" >> .local/collect.md
```

### Step 5: 驗證輸出

```bash
echo "---" >> .local/collect.md
echo "蒐集完成，共 $(grep -c '^-' .local/collect.md) 個項目" >> .local/collect.md
cat .local/collect.md
```

## 完成條件

- `.local/collect.md` 存在且非空。
- 包含時間戳記、檔案清單、磁碟狀態。
