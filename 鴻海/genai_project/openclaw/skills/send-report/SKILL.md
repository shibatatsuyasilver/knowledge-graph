---
name: send-report
description: 讀取整理檔案的報告，並發送給使用者。
metadata:
  openclaw:
    emoji: "📢"
    os: ["darwin", "linux"]
    requires:
      bins: ["cat", "openclaw"]
---

# Send Report Skill

發送工作報告。

## 前置條件

需要先執行 `organize-files` 以產生報告。

## 發送邏輯

```bash
REPORT_FILE="/tmp/organize_report.txt"

if [ -f "$REPORT_FILE" ]; then
    content=$(cat "$REPORT_FILE")
    # 假設 openclaw CLI 支援 send-message 功能
    # 實際環境中可能使用 internal tool "message.send"
    echo "Sending report via OpenClaw..."
    echo "$content"
    # openclaw message send --to "user" --text "$content"
else
    echo "Report file not found. Please run organize-files first."
fi
```
