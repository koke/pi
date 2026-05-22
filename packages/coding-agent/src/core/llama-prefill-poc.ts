import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	buildOpenAICompletionsPayload,
	type Context,
	createAssistantMessageEventStream,
	type Model,
	type SimpleStreamOptions,
	type Usage,
} from "@earendil-works/pi-ai";

type PayloadCallback = NonNullable<SimpleStreamOptions["onPayload"]>;
type ResponseCallback = NonNullable<SimpleStreamOptions["onResponse"]>;

export type DraftPrefillReason = "editor_change" | "tool_result" | "context_change" | "model_change" | "tools_change";

export type LlamaPrefillPocPayloadReason = "launch" | "tool_result_stream" | "tool_result" | "editor_change";

interface LlamaPrefillPocLoggerOptions {
	agentDir: string;
	model: Model<Api>;
	context: Context;
	streamOptions?: SimpleStreamOptions;
}

interface LlamaPrefillPocDraftChangeOptions {
	agentDir: string;
	model: Model<Api>;
	text: string;
	reason: DraftPrefillReason;
	debounceMs: number;
}

interface LlamaPrefillPocPayloadOptions {
	agentDir: string;
	model: Model<Api>;
	context: Context;
	reason: LlamaPrefillPocPayloadReason;
	streamOptions?: SimpleStreamOptions;
	debounceMs?: number;
	chunkChars?: number;
	toolName?: string;
	toolCallId?: string;
}

interface LogFields {
	[key: string]: string | number | boolean | undefined;
}

export class LlamaPrefillPocLogger {
	private readonly logPath: string;
	private readonly model: Model<Api>;
	private readonly requestId = randomUUID();
	private readonly submitAt = Date.now();
	private payloadReadyAt: number | undefined;
	private responseAt: number | undefined;
	private firstTextDeltaAt: number | undefined;

	constructor(options: LlamaPrefillPocLoggerOptions) {
		this.logPath = join(options.agentDir, "llama-prefill-poc.log");
		this.model = options.model;
		this.log("request_start", {
			request_id: this.requestId,
			provider: options.model.provider,
			model: options.model.id,
			api: options.model.api,
			context_chars: stableStringifyLength(options.context),
			context_text_chars: countTextChars(options.context),
			messages: options.context.messages.length,
			tools: options.context.tools?.length ?? 0,
			max_tokens: options.streamOptions?.maxTokens,
		});
	}

	wrapOnPayload(callback: SimpleStreamOptions["onPayload"]): PayloadCallback {
		return async (payload, model) => {
			const beforeCallbackAt = Date.now();
			const nextPayload = await callback?.(payload, model);
			const finalPayload = nextPayload === undefined ? payload : nextPayload;
			const now = Date.now();
			this.payloadReadyAt = now;
			this.log("payload", {
				request_id: this.requestId,
				key: hashPayload(model, finalPayload),
				payload_chars: stableStringifyLength(finalPayload),
				payload_text_chars: countTextChars(finalPayload),
				submit_to_request_ms: now - this.submitAt,
				payload_callback_ms: now - beforeCallbackAt,
				max_tokens: getPayloadMaxTokens(finalPayload),
			});
			return finalPayload;
		};
	}

	wrapOnResponse(callback: SimpleStreamOptions["onResponse"]): ResponseCallback {
		return async (response, model) => {
			const now = Date.now();
			this.responseAt = now;
			this.log("response", {
				request_id: this.requestId,
				status: response.status,
				submit_to_response_ms: now - this.submitAt,
				request_to_first_response_byte_ms:
					this.payloadReadyAt === undefined ? undefined : now - this.payloadReadyAt,
			});
			await callback?.(response, model);
		};
	}

	wrapStream(stream: AssistantMessageEventStream): AssistantMessageEventStream {
		const wrapped = createAssistantMessageEventStream();
		void this.forwardStream(stream, wrapped);
		return wrapped;
	}

	private async forwardStream(
		source: AssistantMessageEventStream,
		target: AssistantMessageEventStream,
	): Promise<void> {
		let sawFinalEvent = false;
		try {
			for await (const event of source) {
				sawFinalEvent ||= event.type === "done" || event.type === "error";
				this.observeEvent(event);
				target.push(event);
			}
			if (!sawFinalEvent) {
				target.push({
					type: "error",
					reason: "error",
					error: this.createErrorMessage("Provider stream ended without a final event"),
				});
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			this.log("stream_error", {
				request_id: this.requestId,
				error: message,
			});
			target.push({
				type: "error",
				reason: "error",
				error: this.createErrorMessage(message),
			});
		}
	}

	private observeEvent(event: AssistantMessageEvent): void {
		if (event.type === "text_delta" && this.firstTextDeltaAt === undefined) {
			const now = Date.now();
			this.firstTextDeltaAt = now;
			this.log("first_text_delta", {
				request_id: this.requestId,
				submit_to_first_visible_token_ms: now - this.submitAt,
				request_to_first_text_delta_ms: this.payloadReadyAt === undefined ? undefined : now - this.payloadReadyAt,
				response_to_first_text_delta_ms: this.responseAt === undefined ? undefined : now - this.responseAt,
			});
			return;
		}
		if (event.type === "done") {
			this.logFinalMessage("request_done", event.message);
			return;
		}
		if (event.type === "error") {
			this.logFinalMessage("request_error", event.error);
		}
	}

	private logFinalMessage(event: "request_done" | "request_error", message: AssistantMessage): void {
		const now = Date.now();
		const usage = message.usage;
		this.log(event, {
			request_id: this.requestId,
			duration_ms: now - this.submitAt,
			submit_to_first_visible_token_ms:
				this.firstTextDeltaAt === undefined ? undefined : this.firstTextDeltaAt - this.submitAt,
			input_tokens: usage.input,
			output_tokens: usage.output,
			cache_read_tokens: usage.cacheRead,
			cache_write_tokens: usage.cacheWrite,
			total_tokens: usage.totalTokens,
			tps: calculateTps(usage, now - this.submitAt),
			stop_reason: message.stopReason,
			prefill_hit: false,
		});
	}

	private createErrorMessage(errorMessage: string): AssistantMessage {
		return {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: this.model.api,
			provider: this.model.provider,
			model: this.model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage,
			timestamp: Date.now(),
		};
	}

	private log(event: string, fields: LogFields): void {
		writeLlamaPrefillPocLog(this.logPath, event, fields);
	}
}

export function createLlamaPrefillPocLogger(options: LlamaPrefillPocLoggerOptions): LlamaPrefillPocLogger | undefined {
	if (options.model.provider !== "llama-cpp") {
		return undefined;
	}
	return new LlamaPrefillPocLogger(options);
}

export function logLlamaPrefillPocDraftChange(options: LlamaPrefillPocDraftChangeOptions): void {
	if (options.model.provider !== "llama-cpp") {
		return;
	}

	const draftText = options.text.trim();
	if (!draftText) {
		return;
	}

	const key = createHash("sha256")
		.update(
			stableStringify({
				provider: options.model.provider,
				model: options.model.id,
				api: options.model.api,
				reason: options.reason,
				text: draftText,
			}),
		)
		.digest("hex")
		.slice(0, 16);

	writeLlamaPrefillPocLog(join(options.agentDir, "llama-prefill-poc.log"), "scheduled", {
		reason: options.reason,
		provider: options.model.provider,
		model: options.model.id,
		api: options.model.api,
		key,
		chars: draftText.length,
		text_chars: countTextChars(draftText),
		debounce_ms: options.debounceMs,
		prefill_active: false,
	});
}

export async function logLlamaPrefillPocPayload(options: LlamaPrefillPocPayloadOptions): Promise<void> {
	if (options.model.provider !== "llama-cpp") {
		return;
	}

	const logPath = join(options.agentDir, "llama-prefill-poc.log");
	if (!isOpenAICompletionsModel(options.model)) {
		writeLlamaPrefillPocLog(logPath, "warmup_payload_skipped", {
			reason: options.reason,
			provider: options.model.provider,
			model: options.model.id,
			api: options.model.api,
			skipped: true,
			skip_reason: "unsupported_api",
			network_sent: false,
		});
		return;
	}

	try {
		const payload = buildOpenAICompletionsPayload(options.model, options.context, options.streamOptions);
		const beforeCallbackAt = Date.now();
		const nextPayload = await options.streamOptions?.onPayload?.(payload, options.model);
		const finalPayload = nextPayload === undefined ? payload : nextPayload;
		const promptPayload = getPromptComparablePayload(finalPayload);
		const now = Date.now();
		writeLlamaPrefillPocLog(logPath, "warmup_payload", {
			reason: options.reason,
			provider: options.model.provider,
			model: options.model.id,
			api: options.model.api,
			key: hashPayload(options.model, finalPayload),
			prompt_key: hashPayload(options.model, promptPayload),
			payload_chars: stableStringifyLength(finalPayload),
			payload_text_chars: countTextChars(finalPayload),
			prompt_chars: stableStringifyLength(promptPayload),
			prompt_text_chars: countTextChars(promptPayload),
			context_chars: stableStringifyLength(options.context),
			context_text_chars: countTextChars(options.context),
			messages: options.context.messages.length,
			tools: options.context.tools?.length ?? 0,
			max_tokens: getPayloadMaxTokens(finalPayload),
			debounce_ms: options.debounceMs,
			chunk_chars: options.chunkChars,
			tool_name: options.toolName,
			tool_call_id: options.toolCallId,
			payload_callback_ms: now - beforeCallbackAt,
			prefill_active: false,
			network_sent: false,
		});
	} catch (error) {
		writeLlamaPrefillPocLog(logPath, "warmup_payload_error", {
			reason: options.reason,
			provider: options.model.provider,
			model: options.model.id,
			api: options.model.api,
			error: error instanceof Error ? error.message : String(error),
			prefill_active: false,
			network_sent: false,
		});
	}
}

function writeLlamaPrefillPocLog(logPath: string, event: string, fields: LogFields): void {
	try {
		mkdirSync(dirname(logPath), { recursive: true });
		appendFileSync(logPath, `${formatLogLine(event, fields)}\n`, "utf8");
	} catch {
		// Best-effort PoC logging must not affect provider requests.
	}
}

function formatLogLine(event: string, fields: LogFields): string {
	const parts = ["llama_prefill", `event=${event}`];
	for (const [key, value] of Object.entries(fields)) {
		if (value === undefined) {
			continue;
		}
		parts.push(`${key}=${formatLogValue(value)}`);
	}
	return parts.join(" ");
}

function formatLogValue(value: string | number | boolean): string {
	if (typeof value === "number" || typeof value === "boolean") {
		return String(value);
	}
	if (/^[A-Za-z0-9._:/@+-]+$/.test(value)) {
		return value;
	}
	return JSON.stringify(value);
}

function hashPayload(model: Model<Api>, payload: unknown): string {
	return createHash("sha256")
		.update(
			stableStringify({
				provider: model.provider,
				model: model.id,
				api: model.api,
				payload,
			}),
		)
		.digest("hex")
		.slice(0, 16);
}

function getPromptComparablePayload(payload: unknown): unknown {
	if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
		return payload;
	}
	const record = payload as Record<string, unknown>;
	return {
		chat_template_kwargs: record.chat_template_kwargs,
		enable_thinking: record.enable_thinking,
		messages: record.messages,
		model: record.model,
		reasoning_effort: record.reasoning_effort,
		thinking: record.thinking,
		tool_choice: record.tool_choice,
		tools: record.tools,
	};
}

function isOpenAICompletionsModel(model: Model<Api>): model is Model<"openai-completions"> {
	return model.api === "openai-completions";
}

function stableStringifyLength(value: unknown): number {
	return stableStringify(value).length;
}

function stableStringify(value: unknown): string {
	if (value === undefined) {
		return "undefined";
	}
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value) ?? String(value);
	}
	if (Array.isArray(value)) {
		return `[${value.map((entry) => stableStringify(entry)).join(",")}]`;
	}
	const record = value as Record<string, unknown>;
	return `{${Object.keys(record)
		.sort()
		.map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
		.join(",")}}`;
}

function countTextChars(value: unknown): number {
	if (typeof value === "string") {
		return value.length;
	}
	if (value === null || typeof value !== "object") {
		return 0;
	}
	if (Array.isArray(value)) {
		return value.reduce<number>((sum, entry) => sum + countTextChars(entry), 0);
	}
	return Object.values(value as Record<string, unknown>).reduce<number>(
		(sum, entry) => sum + countTextChars(entry),
		0,
	);
}

function getPayloadMaxTokens(payload: unknown): number | undefined {
	if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
		return undefined;
	}
	const record = payload as Record<string, unknown>;
	const value = record.max_tokens ?? record.max_completion_tokens;
	return typeof value === "number" ? value : undefined;
}

function calculateTps(usage: Usage, durationMs: number): number | undefined {
	if (usage.output <= 0 || durationMs <= 0) {
		return undefined;
	}
	return Math.round((usage.output / (durationMs / 1000)) * 10) / 10;
}
