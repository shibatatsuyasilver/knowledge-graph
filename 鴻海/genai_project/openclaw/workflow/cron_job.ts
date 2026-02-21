// daily_report_cron.ts
// OpenClaw cron workflow example for multi-skill automation.

export type CronJob = {
  name: string;
  schedule: string;
  agentId: string;
  agentMessage: string;
  concurrent?: boolean;
};

/**
 * Daily report workflow (09:00 every day)
 * 1) /daily-collect
 * 2) /daily-summarize
 * 3) /daily-report
 */
export const dailyReportWorkflowJob: CronJob = {
  name: "daily-report-workflow",
  schedule: "0 9 * * *",
  agentId: "main",
  agentMessage:
    "請依序執行 /daily-collect、/daily-summarize、/daily-report。完成後通知我報告路徑。",
  concurrent: false,
};
