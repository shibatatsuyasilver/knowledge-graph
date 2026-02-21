---
name: daily-report
description: 整合 collect 與 summaries 結果，產生最終每日報告並通知使用者。
metadata: {"openclaw":{"emoji":"📊","os":["darwin","linux"],"requires":{"bins":["cat"]}}}
---

# Daily Report

整合前兩個步驟輸出，產生最終報告與通知。

## 前置條件

```bash
ls -la .local/collect.md .local/summaries.md
if [ $? -ne 0 ]; then
  echo "錯誤：請先執行 /daily-collect 和 /daily-summarize"
  exit 1
fi
```

## 執行步驟

### Step 1: 載入前置輸出

```bash
echo "=== 蒐集報告 ==="
cat .local/collect.md

echo ""
echo "=== 摘要報告 ==="
cat .local/summaries.md
```

### Step 2: 產生最終報告

請根據 collect + summaries 內容輸出以下結構：

- 系統狀態摘要
- 今日文件異動摘要
- 關鍵數據
- 建議行動項目
- 風險提醒

### Step 3: 儲存報告

```bash
REPORT_FILE=".local/daily-report-$(date +%Y-%m-%d).md"
# Agent 會把最終內容寫入此檔案
echo "# 每日工作報告" > "$REPORT_FILE"
```

### Step 4: 通知使用者

```bash
echo "每日報告已產生，請查看 $REPORT_FILE"
# 實際環境可改為 message send 或通道工具通知
```

## 完成條件

- 最終報告檔案存在且包含上述區塊。
- 使用者收到可讀通知。
