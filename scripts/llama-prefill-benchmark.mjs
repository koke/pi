#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";

const DEFAULT_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions";
const DEFAULT_MODEL = "local-model";
const DEFAULT_OUT = "/private/tmp/pi-dev/agent/llama-prefill-benchmark.jsonl";
const TARGET_MODULE = "parser/cache-index.ts";
const EXPECTED_TARGET = `BENCH_TARGET: ${TARGET_MODULE}`;

const TOOL_SIZES = {
	small: { sizeBytes: 4096, chunkBytes: 1024, delayMs: 100 },
	medium: { sizeBytes: 32768, chunkBytes: 4096, delayMs: 100 },
	large: { sizeBytes: 131072, chunkBytes: 8192, delayMs: 100 },
};

const LAUNCH_VARIANTS = ["launch_none", "launch_stable_prefix"];
const TOOL_VARIANTS = ["no_warmup", "tool_result_stream", "tool_result_final", "tool_result_stream_plus_final"];
const EDITOR_VARIANTS = [
	"editor_exact_pause",
	"editor_exact_immediate",
	"editor_tail_typo_corrected",
	"editor_early_typo_corrected",
	"editor_multi_draft",
];

function usage() {
	console.log(`Usage: node scripts/llama-prefill-benchmark.mjs <command> [options]

Commands:
  matrix                         Print benchmark cases
  fixture [--size name]           Print generated tool output fixture
  run                             Run selected direct llama-server benchmark cases
  summarize <jsonl>               Summarize benchmark JSONL output

Options:
  --endpoint <url>                Chat completions endpoint, default ${DEFAULT_ENDPOINT}
  --model <id>                    Model id, default ${DEFAULT_MODEL}
  --out <path>                    JSONL output path, default ${DEFAULT_OUT}
  --scenario <launch|tool|editor> Filter run/matrix scenario
  --variant <name>                Filter run/matrix variant
  --size <small|medium|large>     Tool fixture size, default medium
  --iterations <n>                Run iterations, default 1
  --json                          Print JSON for matrix/summary
  -h, --help                      Show this help
`);
}

function parseArgs(argv) {
	const options = {
		endpoint: DEFAULT_ENDPOINT,
		model: DEFAULT_MODEL,
		out: DEFAULT_OUT,
		scenario: undefined,
		variant: undefined,
		size: "medium",
		sizeFilter: false,
		iterations: 1,
		json: false,
	};
	const positional = [];
	for (let index = 0; index < argv.length; index += 1) {
		const arg = argv[index];
		if (arg === "--help" || arg === "-h") {
			usage();
			process.exit(0);
		}
		if (arg === "--json") {
			options.json = true;
			continue;
		}
		if (arg === "--endpoint") {
			options.endpoint = requireValue(argv, ++index, arg);
			continue;
		}
		if (arg === "--model") {
			options.model = requireValue(argv, ++index, arg);
			continue;
		}
		if (arg === "--out") {
			options.out = requireValue(argv, ++index, arg);
			continue;
		}
		if (arg === "--scenario") {
			options.scenario = requireValue(argv, ++index, arg);
			continue;
		}
		if (arg === "--variant") {
			options.variant = requireValue(argv, ++index, arg);
			continue;
		}
		if (arg === "--size") {
			options.size = requireValue(argv, ++index, arg);
			options.sizeFilter = true;
			continue;
		}
		if (arg === "--iterations") {
			const value = Number.parseInt(requireValue(argv, ++index, arg), 10);
			if (!Number.isFinite(value) || value <= 0) {
				throw new Error("--iterations must be a positive integer");
			}
			options.iterations = value;
			continue;
		}
		positional.push(arg);
	}
	return { command: positional[0], commandArgs: positional.slice(1), options };
}

function requireValue(argv, index, flag) {
	const value = argv[index];
	if (!value) {
		throw new Error(`${flag} requires a value`);
	}
	return value;
}

function benchmarkMatrix(options) {
	const cases = [];
	if (!options.scenario || options.scenario === "launch") {
		for (const variant of LAUNCH_VARIANTS) {
			cases.push({ scenario: "launch", variant, size: undefined, expected: "LAUNCH_OK" });
		}
	}
	if (!options.scenario || options.scenario === "tool") {
		for (const [size, config] of Object.entries(TOOL_SIZES)) {
			for (const variant of TOOL_VARIANTS) {
				cases.push({
					scenario: "tool",
					variant,
					size,
					sizeBytes: config.sizeBytes,
					chunkBytes: config.chunkBytes,
					delayMs: config.delayMs,
					expected: EXPECTED_TARGET,
				});
			}
		}
	}
	if (!options.scenario || options.scenario === "editor") {
		for (const variant of EDITOR_VARIANTS) {
			cases.push({ scenario: "editor", variant, size: undefined, expected: EXPECTED_TARGET });
		}
	}
	return cases.filter((entry) => {
		if (options.variant && entry.variant !== options.variant) {
			return false;
		}
		if (options.sizeFilter && entry.size && entry.size !== options.size) {
			return false;
		}
		return true;
	});
}

function printMatrix(options) {
	const cases = benchmarkMatrix(options);
	if (options.json) {
		console.log(JSON.stringify(cases, null, 2));
		return;
	}
	for (const entry of cases) {
		const size = entry.size ? ` size=${entry.size}` : "";
		console.log(`${entry.scenario}\t${entry.variant}${size}\texpected=${entry.expected}`);
	}
}

function stableSystemPrompt() {
	const lines = [
		"You are Pi benchmark assistant.",
		"Follow exact reply instructions. Do not add explanation.",
		"Benchmark context starts below.",
	];
	for (let index = 0; index < 180; index += 1) {
		const moduleName = `module-${String(index).padStart(3, "0")}.ts`;
		lines.push(
			`Context ${String(index).padStart(3, "0")}: ${moduleName} owns deterministic fixture data, cache-prefix analysis, and terminal workflow notes for llama prefill benchmarking.`,
		);
	}
	lines.push("Benchmark context ends.");
	return lines.join("\n");
}

function benchLogTool() {
	return {
		type: "function",
		function: {
			name: "bench_log",
			description: "Emit deterministic synthetic failure logs for prefill benchmarking.",
			parameters: {
				type: "object",
				properties: {
					case: { type: "string", enum: ["cache-index-failure"] },
					size_bytes: { type: "integer" },
					chunk_bytes: { type: "integer" },
					delay_ms: { type: "integer" },
				},
				required: ["case", "size_bytes", "chunk_bytes", "delay_ms"],
			},
		},
	};
}

function generateFailureLog(sizeBytes) {
	const targetLine = `[001337] ERROR ${TARGET_MODULE} invariant failed: stale prefix map`;
	const lines = [];
	let index = 1;
	while (Buffer.byteLength(`${lines.join("\n")}\n${targetLine}\n`, "utf8") < sizeBytes) {
		const padded = String(index).padStart(6, "0");
		const moduleName = index % 11 === 0 ? "tui/editor.ts" : index % 7 === 0 ? "auth/session.ts" : "agent/loop.ts";
		lines.push(
			`[${padded}] scanning module ${moduleName} ok checksum=${String((index * 7919) % 100000).padStart(5, "0")}`,
		);
		index += 1;
		if (index === 180) {
			lines.push(targetLine);
		}
	}
	if (!lines.includes(targetLine)) {
		const insertAt = Math.floor(lines.length / 2);
		lines.splice(insertAt, 0, targetLine);
	}
	lines.push("Final instruction: identify the module path from the ERROR line.");
	let text = lines.join("\n");
	while (Buffer.byteLength(text, "utf8") < sizeBytes) {
		text += `\n[pad-${Buffer.byteLength(text, "utf8")}] scanning module noop/padding.ts ok`;
	}
	return text;
}

function chunkTextByBytes(text, chunkBytes) {
	const chunks = [];
	let current = "";
	for (const char of text) {
		const next = current + char;
		if (current && Buffer.byteLength(next, "utf8") > chunkBytes) {
			chunks.push(current);
			current = char;
		} else {
			current = next;
		}
	}
	if (current) {
		chunks.push(current);
	}
	return chunks;
}

function launchMessages(userText) {
	return [
		{ role: "system", content: stableSystemPrompt() },
		{ role: "user", content: userText },
	];
}

function toolMessages(logText, sizeConfig) {
	const args = {
		case: "cache-index-failure",
		size_bytes: sizeConfig.sizeBytes,
		chunk_bytes: sizeConfig.chunkBytes,
		delay_ms: sizeConfig.delayMs,
	};
	return [
		{ role: "system", content: stableSystemPrompt() },
		{
			role: "user",
			content: "Analyze this generated failure log and answer with the single failing module name. Reply exactly: BENCH_TARGET: <path>",
		},
		{
			role: "assistant",
			content: "",
			tool_calls: [
				{
					id: "call_bench_log",
					type: "function",
					function: { name: "bench_log", arguments: JSON.stringify(args) },
				},
			],
		},
		{ role: "tool", tool_call_id: "call_bench_log", content: logText },
	];
}

function editorPrompt(variant) {
	const finalPrompt = "Analyze the generated failure log and answer with BENCH_TARGET plus the failing module path.";
	if (variant === "editor_tail_typo_corrected") {
		return {
			finalPrompt,
			warmups: ["Analyze the generated failure log and answer with BENCH_TARGET plus the failing module pth."],
		};
	}
	if (variant === "editor_early_typo_corrected") {
		return {
			finalPrompt,
			warmups: ["Anylaze the generated failure log and answer with BENCH_TARGET plus the failing module path."],
		};
	}
	if (variant === "editor_multi_draft") {
		return {
			finalPrompt,
			warmups: [
				"Analyze the generated failure log.",
				"Analyze the generated failure log and answer with the module.",
				finalPrompt,
			],
		};
	}
	if (variant === "editor_exact_pause") {
		return { finalPrompt, warmups: [finalPrompt] };
	}
	return { finalPrompt, warmups: [] };
}

function requestBody(options, messages, requestOptions) {
	return {
		model: options.model,
		messages,
		tools: requestOptions.includeTools ? [benchLogTool()] : undefined,
		stream: requestOptions.stream,
		max_tokens: requestOptions.maxTokens,
		temperature: 0,
		cache_prompt: true,
	};
}

function pruneUndefined(value) {
	if (Array.isArray(value)) {
		return value.map((entry) => pruneUndefined(entry));
	}
	if (value && typeof value === "object") {
		const record = {};
		for (const [key, entry] of Object.entries(value)) {
			if (entry !== undefined) {
				record[key] = pruneUndefined(entry);
			}
		}
		return record;
	}
	return value;
}

async function runBenchmark(options) {
	if (!TOOL_SIZES[options.size]) {
		throw new Error(`Unknown size: ${options.size}`);
	}
	const cases = benchmarkMatrix(options).filter((entry) => entry.size === undefined || entry.size === options.size);
	if (cases.length === 0) {
		throw new Error("No benchmark cases selected");
	}
	mkdirSync(dirname(options.out), { recursive: true });
	const runId = randomUUID();
	for (let iteration = 0; iteration < options.iterations; iteration += 1) {
		for (const testCase of cases) {
			await runCase({ runId, iteration, testCase, options });
		}
	}
	console.log(`Wrote ${options.out}`);
}

async function runCase({ runId, iteration, testCase, options }) {
	if (testCase.scenario === "launch") {
		await runLaunchCase({ runId, iteration, testCase, options });
		return;
	}
	if (testCase.scenario === "tool") {
		await runToolCase({ runId, iteration, testCase, options });
		return;
	}
	if (testCase.scenario === "editor") {
		await runEditorCase({ runId, iteration, testCase, options });
		return;
	}
	throw new Error(`Unknown scenario: ${testCase.scenario}`);
}

async function runLaunchCase({ runId, iteration, testCase, options }) {
	if (testCase.variant === "launch_stable_prefix") {
		await sendRequest({
			runId,
			iteration,
			testCase,
			options,
			kind: "warmup",
			reason: "launch",
			messages: launchMessages(" "),
			expected: undefined,
			includeTools: true,
			maxTokens: 1,
			stream: false,
		});
	}
	await sendRequest({
		runId,
		iteration,
		testCase,
		options,
		kind: "final",
		reason: undefined,
		messages: launchMessages("Reply exactly: LAUNCH_OK"),
		expected: "LAUNCH_OK",
		includeTools: true,
		maxTokens: 8,
		stream: true,
	});
}

async function runToolCase({ runId, iteration, testCase, options }) {
	const sizeConfig = TOOL_SIZES[testCase.size];
	const logText = generateFailureLog(sizeConfig.sizeBytes);
	const chunks = chunkTextByBytes(logText, sizeConfig.chunkBytes);
	if (testCase.variant === "tool_result_stream" || testCase.variant === "tool_result_stream_plus_final") {
		let partial = "";
		for (const chunk of chunks) {
			partial += chunk;
			await sendRequest({
				runId,
				iteration,
				testCase,
				options,
				kind: "warmup",
				reason: "tool_result_stream",
				messages: toolMessages(partial, sizeConfig),
				expected: undefined,
				includeTools: true,
				maxTokens: 1,
				stream: false,
				extra: { chunk_bytes: Buffer.byteLength(chunk, "utf8"), partial_bytes: Buffer.byteLength(partial, "utf8") },
			});
		}
	}
	if (testCase.variant === "tool_result_final" || testCase.variant === "tool_result_stream_plus_final") {
		await sendRequest({
			runId,
			iteration,
			testCase,
			options,
			kind: "warmup",
			reason: "tool_result",
			messages: toolMessages(logText, sizeConfig),
			expected: undefined,
			includeTools: true,
			maxTokens: 1,
			stream: false,
		});
	}
	await sendRequest({
		runId,
		iteration,
		testCase,
		options,
		kind: "final",
		reason: undefined,
		messages: toolMessages(logText, sizeConfig),
		expected: EXPECTED_TARGET,
		includeTools: true,
		maxTokens: 16,
		stream: true,
	});
}

async function runEditorCase({ runId, iteration, testCase, options }) {
	const prompt = editorPrompt(testCase.variant);
	for (const warmupPrompt of prompt.warmups) {
		await sendRequest({
			runId,
			iteration,
			testCase,
			options,
			kind: "warmup",
			reason: "editor_change",
			messages: launchMessages(warmupPrompt),
			expected: undefined,
			includeTools: true,
			maxTokens: 1,
			stream: false,
		});
	}
	await sendRequest({
		runId,
		iteration,
		testCase,
		options,
		kind: "final",
		reason: undefined,
		messages: launchMessages(prompt.finalPrompt),
		expected: EXPECTED_TARGET,
		includeTools: true,
		maxTokens: 16,
		stream: true,
	});
}

async function sendRequest({
	runId,
	iteration,
	testCase,
	options,
	kind,
	reason,
	messages,
	expected,
	includeTools,
	maxTokens,
	stream,
	extra = {},
}) {
	const body = pruneUndefined(requestBody(options, messages, { includeTools, maxTokens, stream }));
	const payload = stableStringify(body);
	const startedAt = performance.now();
	const response = await fetch(options.endpoint, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
	});
	const responseAt = performance.now();
	let firstTextDeltaMs;
	let content = "";
	let responseText = "";
	if (stream) {
		const streamResult = await readStreamingContent(response, responseAt);
		firstTextDeltaMs = streamResult.firstTextDeltaMs;
		content = streamResult.content;
		responseText = streamResult.responseText;
	} else {
		responseText = await response.text();
		content = parseNonStreamingContent(responseText);
	}
	const endedAt = performance.now();
	const record = {
		ts: new Date().toISOString(),
		run_id: runId,
		iteration,
		scenario: testCase.scenario,
		variant: testCase.variant,
		size: testCase.size,
		kind,
		reason,
		model: options.model,
		status: response.status,
		ok: response.ok,
		payload_key: hashText(payload),
		payload_chars: payload.length,
		payload_text_chars: countTextChars(body),
		duration_ms: Math.round(endedAt - startedAt),
		request_to_first_byte_ms: Math.round(responseAt - startedAt),
		request_to_first_text_delta_ms: firstTextDeltaMs === undefined ? undefined : Math.round(firstTextDeltaMs),
		expected,
		expected_ok: expected === undefined ? undefined : content.trim() === expected,
		content: content.slice(0, 240),
		error_excerpt: response.ok ? undefined : responseText.slice(0, 500),
		...extra,
	};
	appendFileSync(options.out, `${JSON.stringify(pruneUndefined(record))}\n`, "utf8");
	console.log(
		[
			record.scenario,
			record.variant,
			record.size ?? "-",
			record.kind,
			record.status,
			`${record.duration_ms}ms`,
			record.request_to_first_text_delta_ms === undefined ? "first_delta=-" : `first_delta=${record.request_to_first_text_delta_ms}ms`,
			record.expected_ok === undefined ? "" : `expected_ok=${record.expected_ok}`,
		]
			.filter(Boolean)
			.join("\t"),
	);
}

async function readStreamingContent(response, responseAt) {
	if (!response.body) {
		return { content: "", responseText: "", firstTextDeltaMs: undefined };
	}
	const decoder = new TextDecoder();
	const reader = response.body.getReader();
	let buffer = "";
	let responseText = "";
	let content = "";
	let firstTextDeltaMs;
	while (true) {
		const { done, value } = await reader.read();
		if (done) {
			break;
		}
		const chunk = decoder.decode(value, { stream: true });
		responseText += chunk;
		buffer += chunk;
		let newlineIndex = buffer.indexOf("\n");
		while (newlineIndex >= 0) {
			const line = buffer.slice(0, newlineIndex).trimEnd();
			buffer = buffer.slice(newlineIndex + 1);
			const data = line.startsWith("data:") ? line.slice("data:".length).trim() : undefined;
			if (data && data !== "[DONE]") {
				const delta = parseStreamDelta(data);
				if (delta) {
					if (firstTextDeltaMs === undefined) {
						firstTextDeltaMs = performance.now() - responseAt;
					}
					content += delta;
				}
			}
			newlineIndex = buffer.indexOf("\n");
		}
	}
	responseText += decoder.decode();
	return { content, responseText, firstTextDeltaMs };
}

function parseStreamDelta(data) {
	try {
		const parsed = JSON.parse(data);
		const delta = parsed.choices?.[0]?.delta;
		return typeof delta?.content === "string" ? delta.content : "";
	} catch {
		return "";
	}
}

function parseNonStreamingContent(text) {
	try {
		const parsed = JSON.parse(text);
		const content = parsed.choices?.[0]?.message?.content;
		return typeof content === "string" ? content : "";
	} catch {
		return "";
	}
}

function hashText(text) {
	return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

function stableStringify(value) {
	if (value === undefined) {
		return "undefined";
	}
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value) ?? String(value);
	}
	if (Array.isArray(value)) {
		return `[${value.map((entry) => stableStringify(entry)).join(",")}]`;
	}
	return `{${Object.keys(value)
		.sort()
		.map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
		.join(",")}}`;
}

function countTextChars(value) {
	if (typeof value === "string") {
		return value.length;
	}
	if (value === null || typeof value !== "object") {
		return 0;
	}
	if (Array.isArray(value)) {
		return value.reduce((sum, entry) => sum + countTextChars(entry), 0);
	}
	return Object.values(value).reduce((sum, entry) => sum + countTextChars(entry), 0);
}

function summarize(path, options) {
	const lines = readFileSync(path, "utf8")
		.split("\n")
		.filter((line) => line.trim());
	const records = lines.map((line) => JSON.parse(line));
	const groups = new Map();
	for (const record of records) {
		if (record.kind !== "final") {
			continue;
		}
		const key = [record.scenario, record.variant, record.size ?? "-"].join("\t");
		const list = groups.get(key) ?? [];
		list.push(record);
		groups.set(key, list);
	}
	const summary = [];
	for (const [key, recordsForKey] of groups) {
		const [scenario, variant, size] = key.split("\t");
		const firstDeltas = recordsForKey
			.map((record) => record.request_to_first_text_delta_ms)
			.filter((value) => Number.isFinite(value));
		const durations = recordsForKey.map((record) => record.duration_ms).filter((value) => Number.isFinite(value));
		summary.push({
			scenario,
			variant,
			size: size === "-" ? undefined : size,
			runs: recordsForKey.length,
			first_delta_median_ms: quantile(firstDeltas, 0.5),
			first_delta_p90_ms: quantile(firstDeltas, 0.9),
			duration_median_ms: quantile(durations, 0.5),
			expected_ok: recordsForKey.filter((record) => record.expected_ok === true).length,
		});
	}
	if (options.json) {
		console.log(JSON.stringify(summary, null, 2));
		return;
	}
	for (const entry of summary) {
		console.log(
			[
				entry.scenario,
				entry.variant,
				entry.size ?? "-",
				`runs=${entry.runs}`,
				`first_delta_med=${formatMetric(entry.first_delta_median_ms)}`,
				`first_delta_p90=${formatMetric(entry.first_delta_p90_ms)}`,
				`duration_med=${formatMetric(entry.duration_median_ms)}`,
				`expected_ok=${entry.expected_ok}/${entry.runs}`,
			].join("\t"),
		);
	}
}

function quantile(values, q) {
	const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
	if (sorted.length === 0) {
		return undefined;
	}
	if (sorted.length === 1) {
		return sorted[0];
	}
	const position = (sorted.length - 1) * q;
	const lower = Math.floor(position);
	const upper = Math.ceil(position);
	if (lower === upper) {
		return sorted[lower];
	}
	const weight = position - lower;
	return Math.round(sorted[lower] * (1 - weight) + sorted[upper] * weight);
}

function formatMetric(value) {
	return value === undefined ? "n/a" : `${value}ms`;
}

async function main() {
	const { command, commandArgs, options } = parseArgs(process.argv.slice(2));
	if (!command) {
		usage();
		process.exit(1);
	}
	if (command === "matrix") {
		printMatrix(options);
		return;
	}
	if (command === "fixture") {
		const sizeConfig = TOOL_SIZES[options.size];
		if (!sizeConfig) {
			throw new Error(`Unknown size: ${options.size}`);
		}
		process.stdout.write(`${generateFailureLog(sizeConfig.sizeBytes)}\n`);
		return;
	}
	if (command === "run") {
		await runBenchmark(options);
		return;
	}
	if (command === "summarize") {
		const path = commandArgs[0];
		if (!path) {
			throw new Error("summarize requires a JSONL path");
		}
		summarize(path, options);
		return;
	}
	throw new Error(`Unknown command: ${command}`);
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
