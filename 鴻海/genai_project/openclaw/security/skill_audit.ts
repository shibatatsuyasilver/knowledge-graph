import fs from "node:fs";

export type RiskLevel = "low" | "medium" | "high" | "critical";

export interface SecurityRisk {
  id: string;
  level: RiskLevel;
  pattern: string;
  message: string;
}

export interface SecurityAuditResult {
  score: number;
  level: RiskLevel;
  risks: SecurityRisk[];
}

const DANGEROUS_PATTERNS: Array<{
  id: string;
  level: RiskLevel;
  regex: RegExp;
  message: string;
}> = [
  {
    id: "pipe_remote_shell",
    level: "critical",
    regex: /curl\s+[^\n]*\|\s*(bash|sh)\b/i,
    message: "Detected remote script execution via curl | bash.",
  },
  {
    id: "wget_pipe_shell",
    level: "critical",
    regex: /wget\s+[^\n]*-O\s*-\s*\|\s*(bash|sh)\b/i,
    message: "Detected remote script execution via wget -O- | bash.",
  },
  {
    id: "eval_usage",
    level: "high",
    regex: /\beval\s*\(/i,
    message: "Detected eval() usage; this can execute dynamic payloads.",
  },
  {
    id: "base64_decode",
    level: "high",
    regex: /base64\s+(-d|--decode)\b/i,
    message: "Detected base64 decode, often used to hide payloads.",
  },
  {
    id: "sudo_usage",
    level: "high",
    regex: /\bsudo\b/i,
    message: "Detected sudo usage (privilege escalation risk).",
  },
  {
    id: "elevated_true",
    level: "medium",
    regex: /elevated\s*:\s*true/i,
    message: "Skill asks elevated host execution.",
  },
  {
    id: "ssh_scp_rsync",
    level: "medium",
    regex: /\b(ssh|scp|rsync)\b/i,
    message: "Skill invokes remote copy/login tools.",
  },
];

function levelWeight(level: RiskLevel): number {
  if (level === "critical") return 40;
  if (level === "high") return 20;
  if (level === "medium") return 8;
  return 3;
}

function scoreToLevel(score: number): RiskLevel {
  if (score >= 70) return "critical";
  if (score >= 35) return "high";
  if (score >= 12) return "medium";
  return "low";
}

export function auditSkillContent(skillContent: string): SecurityAuditResult {
  const risks: SecurityRisk[] = [];

  for (const check of DANGEROUS_PATTERNS) {
    if (check.regex.test(skillContent)) {
      risks.push({
        id: check.id,
        level: check.level,
        pattern: String(check.regex),
        message: check.message,
      });
    }
  }

  const score = risks.reduce((acc, risk) => acc + levelWeight(risk.level), 0);
  const level = scoreToLevel(score);

  return {
    score,
    level,
    risks,
  };
}

export function auditSkillFile(filePath: string): SecurityAuditResult {
  const content = fs.readFileSync(filePath, "utf8");
  return auditSkillContent(content);
}

if (require.main === module) {
  const filePath = process.argv[2];
  if (!filePath) {
    console.error("Usage: node skill_audit.ts <SKILL.md>");
    process.exit(1);
  }

  const result = auditSkillFile(filePath);
  console.log(JSON.stringify(result, null, 2));
}
