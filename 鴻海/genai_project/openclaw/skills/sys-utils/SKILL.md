---
name: sys-utils
description: 一個綜合工具技能，包含查詢系統時間與讀取檔案摘要的功能。
metadata:
  openclaw:
    emoji: "🛠️"
    os: ["darwin", "linux"]
    requires:
      bins: ["date", "cat", "head", "wc"]
---

# System Utilities Skill

提供系統資訊查詢與檔案摘要功能。

## 功能一：查詢時間

當使用者詢問「現在幾點」、「系統時間」時使用。

```bash
# 顯示本地時間與 UTC 時間
echo "Local: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "UTC:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
```

## 功能二：檔案摘要

當使用者要求「讀取檔案摘要」、「查看檔案內容」時使用。
請替換 `{{filepath}}` 為實際檔案路徑。

```bash
filepath="{{filepath}}"

if [ -f "$filepath" ]; then
    echo "=== File Info ==="
    ls -lh "$filepath"
    echo "=== Content Preview (First 20 lines) ==="
    head -n 20 "$filepath"
    echo "..."
    echo "=== End of Preview ==="
else
    echo "Error: File not found at $filepath"
fi
```
