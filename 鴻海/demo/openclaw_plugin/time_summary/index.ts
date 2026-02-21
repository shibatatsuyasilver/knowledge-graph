import fs from "node:fs/promises";
import path from "node:path";

import { Type } from "@sinclair/typebox";

const DEFAULT_MAX_READ_BYTES = 16_384;

function parseConfig(raw: unknown) {
  const cfg = raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};

  const allowedRoots = Array.isArray(cfg.allowedRoots)
    ? cfg.allowedRoots.filter((x): x is string => typeof x === "string" && x.trim().length > 0)
    : [process.cwd()];

  const maxReadBytes =
    typeof cfg.maxReadBytes === "number" && cfg.maxReadBytes > 0
      ? Math.floor(cfg.maxReadBytes)
      : DEFAULT_MAX_READ_BYTES;

  const enableReadSummary = typeof cfg.enableReadSummary === "boolean" ? cfg.enableReadSummary : true;

  return {
    allowedRoots: allowedRoots.map((p) => path.resolve(p)),
    maxReadBytes,
    enableReadSummary,
  };
}

function normalizeSentence(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function summarizeText(text: string, maxSentences: number): string {
  const cleaned = text.replace(/\r\n/g, "\n").replace(/\n+/g, " ").trim();
  if (!cleaned) return "(empty file)";

  const sentences = cleaned
    .split(/(?<=[.!?。！？])\s+/)
    .map(normalizeSentence)
    .filter(Boolean);

  if (sentences.length === 0) {
    return cleaned.slice(0, 240);
  }

  return sentences.slice(0, Math.max(1, maxSentences)).join(" ");
}

function assertAllowedPath(inputPath: string, allowedRoots: string[]): string {
  const resolved = path.resolve(inputPath);

  const allowed = allowedRoots.some((root) => {
    const normalizedRoot = root.endsWith(path.sep) ? root : `${root}${path.sep}`;
    return resolved === root || resolved.startsWith(normalizedRoot);
  });

  if (!allowed) {
    throw new Error(`Path not allowed: ${resolved}`);
  }

  return resolved;
}

const plugin = {
  id: "time-summary",
  name: "Time Summary Plugin",
  description: "Provide local time and safe file-summary tools for OpenClaw workflows.",
  configSchema: {
    parse: parseConfig,
    uiHints: {
      allowedRoots: { label: "Allowed Roots", help: "Whitelisted file roots for read_and_summarize." },
      maxReadBytes: { label: "Max Read Bytes" },
      enableReadSummary: { label: "Enable Read Summary" },
    },
  },

  register(api: any) {
    const cfg = parseConfig(api.pluginConfig);

    api.registerTool({
      name: "local_time",
      description: "Return current local time data. Optional timezone override.",
      parameters: Type.Object({
        timezone: Type.Optional(Type.String({ description: "IANA timezone, e.g. Asia/Taipei" })),
      }),
      async execute(_id: string, params: Record<string, unknown>) {
        const timezone =
          typeof params.timezone === "string" && params.timezone.trim().length > 0
            ? params.timezone.trim()
            : Intl.DateTimeFormat().resolvedOptions().timeZone;

        const now = new Date();
        const local = now.toLocaleString("sv-SE", {
          timeZone: timezone,
          hour12: false,
        });

        const result = {
          isoTime: now.toISOString(),
          localTime: local,
          timezone,
          epochMs: now.getTime(),
        };

        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
          details: result,
        };
      },
    });

    api.registerTool(
      {
        name: "read_and_summarize",
        description: "Read a local text file from an allowed root and return a short summary.",
        parameters: Type.Object({
          path: Type.String({ description: "Absolute or relative file path" }),
          maxBytes: Type.Optional(Type.Number({ minimum: 256 })),
          maxSentences: Type.Optional(Type.Number({ minimum: 1, maximum: 12 })),
        }),
        async execute(_id: string, params: Record<string, unknown>) {
          if (!cfg.enableReadSummary) {
            throw new Error("read_and_summarize is disabled by plugin config");
          }

          const inputPath = String(params.path ?? "").trim();
          if (!inputPath) {
            throw new Error("path is required");
          }

          const maxBytesCandidate = typeof params.maxBytes === "number" ? Math.floor(params.maxBytes) : cfg.maxReadBytes;
          const maxBytes = Math.min(Math.max(maxBytesCandidate, 256), cfg.maxReadBytes);

          const maxSentencesCandidate =
            typeof params.maxSentences === "number" ? Math.floor(params.maxSentences) : 3;
          const maxSentences = Math.min(Math.max(maxSentencesCandidate, 1), 12);

          const resolvedPath = assertAllowedPath(inputPath, cfg.allowedRoots);
          const raw = await fs.readFile(resolvedPath, "utf8");
          const clipped = raw.slice(0, maxBytes);
          const summary = summarizeText(clipped, maxSentences);

          const result = {
            sourcePath: resolvedPath,
            usedBytes: Buffer.byteLength(clipped, "utf8"),
            maxBytes,
            summary,
          };

          return {
            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
            details: result,
          };
        },
      },
      { optional: true },
    );

    api.logger?.info?.("[time-summary] plugin registered", {
      roots: cfg.allowedRoots,
      maxReadBytes: cfg.maxReadBytes,
      enableReadSummary: cfg.enableReadSummary,
    });
  },
};

export default plugin;
