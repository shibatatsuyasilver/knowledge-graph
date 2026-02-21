const fs = require("node:fs");

const DANGEROUS_PATTERNS = [
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

function levelWeight(level) {
  if (level === "critical") return 40;
  if (level === "high") return 20;
  if (level === "medium") return 8;
  return 3;
}

function scoreToLevel(score) {
  if (score >= 70) return "critical";
  if (score >= 35) return "high";
  if (score >= 12) return "medium";
  return "low";
}

function auditSkillContent(skillContent) {
  const risks = [];

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
  return {
    score,
    level: scoreToLevel(score),
    risks,
  };
}

function auditSkillFile(filePath) {
  const content = fs.readFileSync(filePath, "utf8");
  return auditSkillContent(content);
}

if (require.main === module) {
  const filePath = process.argv[2];
  if (!filePath) {
    console.error("Usage: node skill_audit.js <SKILL.md>");
    process.exit(1);
  }
  const result = auditSkillFile(filePath);
  console.log(JSON.stringify(result, null, 2));
}

module.exports = {
  auditSkillContent,
  auditSkillFile,
};
