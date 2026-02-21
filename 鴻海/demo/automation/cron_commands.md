# OpenClaw 自動化流程：任務抓取 + 每日摘要

本文件示範一個包含 **兩個以上 skill/tool** 的自動化流程：
- `apple-reminders`：抓取今日待辦
- `time-summary`（自製 plugin）：補上本地時間與檔案摘要能力
- `discord`：將摘要推播到指定 channel

## 1) 前置準備

1. 啟用 OpenClaw gateway 與 Discord channel。
2. macOS 安裝 reminders CLI：

```bash
brew install steipete/tap/remindctl
remindctl authorize
```

3. 啟用遷移後的自製 plugin（位於 `openclaw-main/extensions/time-summary`）：

```bash
openclaw plugins enable time-summary
# 重新啟動 gateway 後生效
```

4. 設定 plugin config（`plugins.entries.time-summary.config`）：

```json5
{
  plugins: {
    entries: {
      "time-summary": {
        enabled: true,
        config: {
          allowedRoots: ["/Users/silver/Documents/openclaw-main/docs/automation"],
          maxReadBytes: 32768,
          enableReadSummary: true
        }
      }
    }
  }
}
```

## 2) 每日排程命令

```bash
openclaw cron add \
  --name "daily-todo-digest" \
  --cron "30 8 * * *" \
  --tz "Asia/Taipei" \
  --session isolated \
  --message "
你是一個自動化助理，請執行以下流程：
1) 呼叫 apple-reminders（remindctl today --json）取得今日待辦。
2) 呼叫 local_time 取得當前時間與時區。
3) 將待辦依 Priority/到期時間分類，生成 5~8 行摘要。
4) 如 /Users/silver/Documents/openclaw-main/docs/automation/context.txt 存在，呼叫 read_and_summarize 附加重點。
5) 以 discord tool 發送最終摘要到目標頻道，並回報發送結果。
若任一步驟失敗，請輸出可讀錯誤與下一步建議。" \
  --deliver \
  --channel discord \
  --to "channel:<DISCORD_CHANNEL_ID>"
```

## 3) 流程邏輯說明

1. **Cron 觸發**：08:30 執行 isolated session，避免汙染主對話。
2. **Skill 1（apple-reminders）**：取回今日待辦資料。
3. **Skill 2（time-summary/local_time）**：加入當前時間戳與時區。
4. **Skill 3（time-summary/read_and_summarize，可選）**：讀取白名單文件並生成補充摘要。
5. **Skill 4（discord）**：推播 digest。
6. **結果回報**：輸出「成功/失敗、訊息 ID 或錯誤原因」。

## 4) 失敗情境與重試策略

- `remindctl` 權限不足：回報「Reminders permission missing」，提示執行 `remindctl authorize`。
- Discord 目標錯誤：回報 channel id 無效，保留摘要於回傳內容，等待人工修正。
- 檔案摘要路徑不合法：`read_and_summarize` 直接拒絕（Path not allowed），不中止主摘要流程。
- 單次暫時失敗：建議重跑 `openclaw cron run <jobId> --force`。

## 5) 驗收檢查點

- 每日固定時間可觸發 job。
- 摘要包含：日期時間、待辦清單、優先級、Discord 投遞結果。
- 至少使用兩個 skill/tool（本流程為三個以上）。
- 錯誤訊息可讀且可指引下一步修復。
