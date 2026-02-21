---
name: system-info
description: 查詢本機系統時間、作業系統資訊、磁碟用量與網路狀態。
metadata: {"openclaw":{"emoji":"🖥️","os":["darwin","linux"],"requires":{"bins":["date"]}}}
---

# System Info Skill

查詢本機系統資訊，包含時間、OS 版本、磁碟用量與網路狀態。

## 觸發條件

- 現在幾點？
- 目前時間？
- 系統狀態如何？
- 磁碟空間還剩多少？
- 伺服器 uptime？

## 時間查詢

```bash
# 查詢目前本地時間（含時區）
date '+%Y-%m-%d %H:%M:%S %Z (UTC%z)'

# 查詢 UTC 時間
date -u '+%Y-%m-%d %H:%M:%S UTC'

# 查詢特定時區時間（例如東京）
TZ='Asia/Tokyo' date '+%Y-%m-%d %H:%M:%S %Z'

# 查詢 Unix timestamp
date +%s
```

## 系統資訊

```bash
# 作業系統版本
uname -srm

# 系統 uptime
uptime

# 主機名稱
hostname
```

## 磁碟用量

```bash
# 根目錄磁碟用量
df -h / | tail -1

# 目前目錄用量
du -sh .
```

## 網路狀態

```bash
# 本機 IP 位址
hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || echo "無法取得 IP"

# 測試外部連線
ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 && echo "網路正常" || echo "網路異常"
```

## 注意事項

- 時間查詢不需要額外安裝套件，使用系統內建 `date`。
- 避免執行長時間網路診斷（例如 traceroute）。
- 磁碟查詢僅顯示摘要資訊。
