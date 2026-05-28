import { closeSync, existsSync, openSync, readSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext, Theme } from "@earendil-works/pi-coding-agent";

type LogFields = Record<string, string>;

const STATUS_KEY = "llama-prefill";
const DEFAULT_AGENT_DIR = join(homedir(), ".pi", "agent");
const POLL_MS = 500;
const MAX_TAIL_BYTES = 64 * 1024;
const SOURCE_PID = String(process.pid);

function logPath(): string {
	return join(process.env.PI_CODING_AGENT_DIR ?? DEFAULT_AGENT_DIR, "llama-prefill-poc.log");
}

function readTail(path: string): string {
	try {
		if (!existsSync(path)) return "";

		const stat = statSync(path);
		const length = Math.min(stat.size, MAX_TAIL_BYTES);
		if (length <= 0) return "";

		const buffer = Buffer.alloc(length);
		const fd = openSync(path, "r");
		try {
			readSync(fd, buffer, 0, length, stat.size - length);
			return buffer.toString("utf8");
		} finally {
			closeSync(fd);
		}
	} catch {
		return "";
	}
}

function parseFields(line: string): LogFields | undefined {
	if (!line.startsWith("llama_prefill ")) return undefined;

	const fields: LogFields = {};
	for (const match of line.matchAll(/([A-Za-z0-9_]+)=("[^"]*"|\S+)/g)) {
		const key = match[1];
		const raw = match[2];
		if (!key || !raw) continue;
		if (raw.startsWith('"')) {
			try {
				const value = JSON.parse(raw) as unknown;
				fields[key] = typeof value === "string" ? value : String(value);
			} catch {
				fields[key] = raw.slice(1, -1);
			}
		} else {
			fields[key] = raw;
		}
	}
	return fields.event ? fields : undefined;
}

function latestWarmupEvent(path: string): LogFields | undefined {
	const lines = readTail(path).split("\n");
	for (let index = lines.length - 1; index >= 0; index -= 1) {
		const fields = parseFields(lines[index].trim());
		if (fields?.event?.startsWith("warmup_") && fields.source_pid === SOURCE_PID) {
			return fields;
		}
	}
	return undefined;
}

function reasonLabel(reason: string | undefined): string {
	if (reason === "editor_change") return "editor";
	if (reason === "tool_result_stream") return "tool stream";
	if (reason === "tool_result") return "tool";
	if (reason === "launch") return "launch";
	if (reason === "context_change") return "context";
	if (reason === "model_change") return "model";
	if (reason === "tools_change") return "tools";
	return "warmup";
}

function formatMs(value: string | undefined): string | undefined {
	const ms = value === undefined ? NaN : Number(value);
	if (!Number.isFinite(ms)) return undefined;
	if (ms < 1000) return `${Math.round(ms)}ms`;
	return `${(ms / 1000).toFixed(1)}s`;
}

function formatStatus(fields: LogFields, theme: Theme): string | undefined {
	const label = reasonLabel(fields.reason);
	if (fields.event === "warmup_payload" || fields.event === "warmup_start") {
		return theme.fg("muted", `draft: ${label} warming`);
	}
	if (fields.event === "warmup_done") {
		const duration = formatMs(fields.duration_ms ?? fields.request_to_response_ms);
		return theme.fg("accent", `draft: ${label} ready${duration ? ` ${duration}` : ""}`);
	}
	if (fields.event === "warmup_aborted") {
		const reason = fields.cancel_reason ? ` ${fields.cancel_reason}` : "";
		return theme.fg("muted", `draft: ${label} canceled${reason}`);
	}
	if (fields.event === "warmup_error" || fields.event === "warmup_payload_error") {
		const error = fields.error ? ` ${fields.error}` : "";
		return theme.fg("warning", `draft: ${label} failed${error}`);
	}
	if (fields.event === "warmup_payload_skipped") {
		const skip = fields.skip_reason ? ` ${fields.skip_reason}` : "";
		return theme.fg("muted", `draft: ${label} skipped${skip}`);
	}
	return undefined;
}

function isPrefillPocModel(ctx: ExtensionContext): boolean {
	return isPrefillPocProvider(ctx.model?.provider) || isLocalOpenAICompatibleBaseUrl(ctx.model?.baseUrl);
}

function isPrefillPocProvider(provider: string | undefined): boolean {
	return provider === "llama-cpp" || provider === "lm-studio" || provider === "lmstudio";
}

function isLocalOpenAICompatibleBaseUrl(baseUrl: string | undefined): boolean {
	if (!baseUrl) return false;
	try {
		const url = new URL(baseUrl);
		return (url.hostname === "127.0.0.1" || url.hostname === "localhost") && url.port === "1234";
	} catch {
		return false;
	}
}

export default function llamaPrefillStatusExtension(pi: ExtensionAPI) {
	if (process.env.PI_LLAMA_PREFILL_STATUS === "0") return;

	let interval: ReturnType<typeof setInterval> | undefined;
	let lastStatus: string | undefined;

	const stop = (ctx?: ExtensionContext) => {
		if (interval) {
			clearInterval(interval);
			interval = undefined;
		}
		lastStatus = undefined;
		ctx?.ui.setStatus(STATUS_KEY, undefined);
	};

	const update = (ctx: ExtensionContext) => {
		if (!ctx.hasUI || !isPrefillPocModel(ctx)) {
			stop(ctx);
			return;
		}
		const fields = latestWarmupEvent(logPath());
		const status = fields ? formatStatus(fields, ctx.ui.theme) : ctx.ui.theme.fg("muted", "draft: idle");
		if (status !== lastStatus) {
			ctx.ui.setStatus(STATUS_KEY, status);
			lastStatus = status;
		}
	};

	const start = (ctx: ExtensionContext) => {
		if (!ctx.hasUI || !isPrefillPocModel(ctx)) {
			stop(ctx);
			return;
		}
		if (interval) clearInterval(interval);
		interval = setInterval(() => update(ctx), POLL_MS);
		update(ctx);
	};

	pi.on("session_start", async (_event, ctx) => {
		start(ctx);
	});

	pi.on("model_select", async (_event, ctx) => {
		start(ctx);
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		stop(ctx);
	});
}
