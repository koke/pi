#!/usr/bin/env node

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

const defaults = {
	session: "pi-poc",
	port: "8081",
	agentDir: "/private/tmp/pi-dev/agent",
	sessionDir: "/private/tmp/pi-dev/sessions",
	cwd: "/Users/koke/src/pi",
	model: "gemma",
};

const models = {
	gemma: {
		path: "/Users/koke/models/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q4_K_XL.gguf",
		log: "/private/tmp/pi-dev/agent/gemma-server.log",
		contextWindow: 32768,
		parallel: 2,
		serverArgs: ["--jinja", "--reasoning", "off"],
	},
	qwen27b: {
		path: "/Users/koke/models/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf",
		log: "/private/tmp/pi-dev/agent/llama-server.log",
		contextWindow: 32768,
		parallel: 1,
		serverArgs: ["--reasoning", "off"],
	},
	qwen35a3b: {
		path: "/Users/koke/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
		log: "/private/tmp/pi-dev/agent/llama-server.log",
		contextWindow: 32768,
		parallel: 1,
		serverArgs: ["--reasoning", "off"],
	},
};

function usage() {
	console.log(`Usage: node scripts/llama-prefill-poc.mjs <command> [options]

Commands:
  status                         Show pi-poc panes
  setup                          Ensure two panes and isolated models.json
  restart-server [--model name]   Restart llama-server in left pane
  start-pi                       Start Pi in right pane
  stop-server                    Send Ctrl-C to left pane
  stop-pi                        Send Ctrl-D to right pane
  smoke [expected text]           Send a smoke prompt to Pi
  capture <server|pi> [lines]     Capture recent pane output
  tail-server [lines]             Tail the selected model server log

Options:
  --session <name>                tmux session, default ${defaults.session}
  --port <port>                   llama-server port, default ${defaults.port}
  --agent-dir <dir>               isolated agent dir, default ${defaults.agentDir}
  --session-dir <dir>             isolated session dir, default ${defaults.sessionDir}
  --model <${Object.keys(models).join("|")}>          model preset, default ${defaults.model}
`);
}

function parseArgs(argv) {
	const options = { ...defaults };
	const positional = [];
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === "--help" || arg === "-h") {
			usage();
			process.exit(0);
		}
		if (arg === "--session" || arg === "--port" || arg === "--agent-dir" || arg === "--session-dir" || arg === "--model") {
			const value = argv[++i];
			if (!value) {
				throw new Error(`${arg} requires a value`);
			}
			if (arg === "--session") options.session = value;
			if (arg === "--port") options.port = value;
			if (arg === "--agent-dir") options.agentDir = value;
			if (arg === "--session-dir") options.sessionDir = value;
			if (arg === "--model") options.model = value;
			continue;
		}
		positional.push(arg);
	}
	return { command: positional[0], commandArgs: positional.slice(1), options };
}

function run(command, args, options = {}) {
	const result = spawnSync(command, args, {
		cwd: options.cwd,
		encoding: "utf8",
		stdio: options.capture ? ["inherit", "pipe", "pipe"] : "inherit",
	});
	if (result.status !== 0) {
		const stderr = result.stderr ? `\n${result.stderr.trim()}` : "";
		throw new Error(`Command failed: ${[command, ...args].join(" ")}${stderr}`);
	}
	return result.stdout ?? "";
}

function tmux(args, options = {}) {
	return run("tmux", args, { capture: options.capture });
}

function tmuxOutput(args) {
	return tmux(args, { capture: true });
}

function firstWindow(session) {
	const output = tmuxOutput(["list-windows", "-t", session, "-F", "#{window_index}"]).trim();
	const index = output.split("\n").find(Boolean);
	if (!index) {
		throw new Error(`No windows found in tmux session ${session}`);
	}
	return `${session}:${index}`;
}

function paneTargets(options) {
	const window = firstWindow(options.session);
	return {
		server: `${window}.0`,
		pi: `${window}.1`,
	};
}

function ensureLayout(options) {
	const window = firstWindow(options.session);
	const output = tmuxOutput(["list-panes", "-t", window, "-F", "#{pane_index}"]).trim();
	const panes = output ? output.split("\n") : [];
	if (panes.length === 1) {
		tmux(["split-window", "-h", "-t", `${window}.0`, "-c", defaults.cwd]);
	}
}

function sendKeys(target, keys) {
	tmux(["send-keys", "-t", target, ...keys]);
}

function writeModelsConfig(options) {
	mkdirSync(options.agentDir, { recursive: true });
	mkdirSync(options.sessionDir, { recursive: true });
	const model = models[options.model];
	if (!model) {
		throw new Error(`Unknown model preset: ${options.model}`);
	}
	const config = {
		providers: {
			"llama-cpp": {
				baseUrl: `http://127.0.0.1:${options.port}/v1`,
				api: "openai-completions",
				apiKey: "llama.cpp",
				compat: {
					supportsDeveloperRole: false,
					supportsReasoningEffort: false,
					supportsUsageInStreaming: false,
					maxTokensField: "max_tokens",
				},
				models: [
					{
						id: "local-model",
						name: "llama.cpp local",
						reasoning: false,
						input: ["text"],
						contextWindow: model.contextWindow,
						maxTokens: 8192,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					},
				],
			},
		},
	};
	writeFileSync(join(options.agentDir, "models.json"), `${JSON.stringify(config, null, 2)}\n`);
}

function shellQuote(value) {
	return `'${value.replaceAll("'", "'\\''")}'`;
}

function restartServer(options) {
	const model = models[options.model];
	if (!model) {
		throw new Error(`Unknown model preset: ${options.model}`);
	}
	if (!existsSync(model.path)) {
		throw new Error(`Model file not found: ${model.path}`);
	}
	writeModelsConfig(options);
	ensureLayout(options);
	const targets = paneTargets(options);
	sendKeys(targets.server, ["C-c"]);
	const command = [
		"llama-server",
		"-m",
		shellQuote(model.path),
		"--host",
		"127.0.0.1",
		"--port",
		options.port,
		"--ctx-size",
		String(model.contextWindow),
		"--parallel",
		String(model.parallel),
		"--slots",
		"--cache-prompt",
		...model.serverArgs,
		"2>&1",
		"|",
		"tee",
		shellQuote(model.log),
	].join(" ");
	sendKeys(targets.server, [command, "Enter"]);
}

function startPi(options) {
	writeModelsConfig(options);
	ensureLayout(options);
	const targets = paneTargets(options);
	const command = [
		`PI_CODING_AGENT_DIR=${shellQuote(options.agentDir)}`,
		`PI_CODING_AGENT_SESSION_DIR=${shellQuote(options.sessionDir)}`,
		"PI_OFFLINE=1",
		"PI_SKIP_VERSION_CHECK=1",
		"./pi-test.sh",
		"--model",
		"llama-cpp/local-model",
	].join(" ");
	sendKeys(targets.pi, [command, "Enter"]);
}

function capturePane(options, paneName, lineCount) {
	ensureLayout(options);
	const targets = paneTargets(options);
	const target = targets[paneName];
	if (!target) {
		throw new Error("capture requires pane name: server or pi");
	}
	const lines = Number.isFinite(lineCount) ? lineCount : 60;
	const output = tmuxOutput(["capture-pane", "-t", target, "-p", "-S", `-${lines}`]);
	process.stdout.write(output);
}

function tailServer(options, lineCount) {
	const model = models[options.model];
	if (!model) {
		throw new Error(`Unknown model preset: ${options.model}`);
	}
	const lines = Number.isFinite(lineCount) ? lineCount : 80;
	const path = model.log;
	if (!existsSync(path)) {
		throw new Error(`Server log not found: ${path}`);
	}
	const content = readFileSync(path, "utf8").split("\n");
	console.log(content.slice(-lines).join("\n"));
}

function smoke(options, text) {
	ensureLayout(options);
	const targets = paneTargets(options);
	const expected = text || "smoke ok";
	sendKeys(targets.pi, [`Reply with exactly: ${expected}`, "Enter"]);
}

function main() {
	const { command, commandArgs, options } = parseArgs(process.argv.slice(2));
	if (!command) {
		usage();
		process.exit(1);
	}
	if (command === "status") {
		process.stdout.write(
			tmuxOutput([
				"list-panes",
				"-t",
				options.session,
				"-a",
				"-F",
				"#S:#I.#P #{pane_id} #{pane_width}x#{pane_height} #{pane_current_command} #{pane_current_path}",
			]),
		);
		return;
	}
	if (command === "setup") {
		ensureLayout(options);
		writeModelsConfig(options);
		return;
	}
	if (command === "restart-server") {
		restartServer(options);
		return;
	}
	if (command === "start-pi") {
		startPi(options);
		return;
	}
	if (command === "stop-server") {
		sendKeys(paneTargets(options).server, ["C-c"]);
		return;
	}
	if (command === "stop-pi") {
		sendKeys(paneTargets(options).pi, ["C-d"]);
		return;
	}
	if (command === "smoke") {
		smoke(options, commandArgs.join(" "));
		return;
	}
	if (command === "capture") {
		capturePane(options, commandArgs[0], Number(commandArgs[1]));
		return;
	}
	if (command === "tail-server") {
		tailServer(options, Number(commandArgs[0]));
		return;
	}
	throw new Error(`Unknown command: ${command}`);
}

try {
	main();
} catch (error) {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
}
