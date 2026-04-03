import { fillNormal, mulberry32 } from "./noise.ts";
import { ort } from "./runtime.ts";

export interface Entry {
	name: string;
	file: string;
	width: number;
	steps: number;
	context: number;
	params: number;
	gflops: number;
}

export interface Result extends Entry {
	medianMs?: number;
	p90Ms?: number;
	minMs?: number;
	ticksPerSecond?: number;
	// Node count per execution provider, from onnxruntime's verbose log
	placement?: Record<string, number>;
	// "OpType (node name)" for every node not on WebGPU
	offGpuNodes?: string[];
	offGpuShapeOps?: string[];
	offGpuComputeOps?: string[];
	error?: string;
}

const frameSize = 64;
const warmupRuns = 3;
const timedRuns = 100;
const noiseSeed = 0;
// onnxruntime puts these on CPU on purpose when they only compute shapes
const shapeOps = new Set([
	"Shape",
	"Size",
	"Reshape",
	"Squeeze",
	"Unsqueeze",
	"Flatten",
	"ConstantOfShape",
]);

// Emscripten binds console.log and console.error when onnxruntime's wasm starts, so the tap must exist before the first session
let captured: string[] | null = null;
for (const method of ["log", "error"] as const) {
	const original = console[method].bind(console);
	console[method] = (...args: unknown[]) => {
		if (captured) captured.push(args.join(" "));
		else original(...args);
	};
}

async function placementLog(bytes: Uint8Array): Promise<string[]> {
	const lines: string[] = [];
	captured = lines;
	try {
		const session = await ort.InferenceSession.create(bytes, {
			executionProviders: ["webgpu"],
			logSeverityLevel: 0,
		});
		await session.release();
	} finally {
		captured = null;
	}
	return lines;
}

// One "placed on [Provider]. Number of nodes: n" line per provider, each followed by one "OpType (name)" line per node unless all nodes share one provider
function parsePlacement(lines: string[]): {
	placement: Record<string, number>;
	offGpuNodes: string[];
} {
	const placement: Record<string, number> = {};
	const offGpuNodes: string[] = [];
	let provider = "";

	for (const line of lines) {
		const header = line.match(/placed on \[(\w+)\]\. Number of nodes: (\d+)$/);
		const node = line.match(/VerifyEachNodeIsAssignedToAnEp\]\s+(\S+ \(.*\))$/);
		if (header) {
			provider = header[1];
			placement[provider] = Number(header[2]);
		} else if (node && provider !== "WebGpuExecutionProvider") {
			offGpuNodes.push(node[1]);
		}
	}

	if (Object.keys(placement).length === 0)
		throw new Error("No node placement lines in the onnxruntime verbose log");
	// Catches a changed log format, and a graph entirely off WebGPU (a single provider gets no per-node lines)
	const offGpuCount = Object.entries(placement).reduce(
		(sum, [name, count]) =>
			name === "WebGpuExecutionProvider" ? sum : sum + count,
		0,
	);
	if (offGpuNodes.length !== offGpuCount) {
		throw new Error(
			`Parsed ${offGpuNodes.length} nodes off WebGPU, but the log reports ${JSON.stringify(placement)}`,
		);
	}
	return { placement, offGpuNodes };
}

// Separate session from placementLog, so verbose logging cannot slow the timed runs
async function time(bytes: Uint8Array, context: number): Promise<number[]> {
	const session = await ort.InferenceSession.create(bytes, {
		executionProviders: ["webgpu"],
	});
	try {
		const noise = new Float32Array(3 * (context + 1) * frameSize * frameSize);
		fillNormal(noise, mulberry32(noiseSeed));
		const feeds = {
			context: new ort.Tensor(
				"float32",
				new Float32Array(3 * context * frameSize * frameSize),
				[1, 3 * context, frameSize, frameSize],
			),
			actions: new ort.Tensor("int64", new BigInt64Array(context), [
				1,
				context,
			]),
			noise: new ort.Tensor("float32", noise, [
				1,
				3 * (context + 1),
				frameSize,
				frameSize,
			]),
		};

		const times: number[] = [];
		for (let i = 0; i < warmupRuns + timedRuns; i++) {
			const start = performance.now();
			const { frame } = await session.run(feeds);
			// Read the frame back to the CPU, as the game loop does every tick
			await frame.getData();
			if (i >= warmupRuns) times.push(performance.now() - start);
		}
		return times;
	} finally {
		await session.release();
	}
}

// Linear interpolation between ranks, as numpy.quantile does
function quantile(sorted: number[], q: number): number {
	const position = (sorted.length - 1) * q;
	const lower = Math.floor(position);
	return (
		sorted[lower] +
		(sorted[Math.ceil(position)] - sorted[lower]) * (position - lower)
	);
}

export async function benchEntry(
	entry: Entry,
	bytes: Uint8Array,
): Promise<Result> {
	const { placement, offGpuNodes } = parsePlacement(await placementLog(bytes));
	const offGpuOps = [...new Set(offGpuNodes.map(node => node.split(" ")[0]))];

	const sorted = (await time(bytes, entry.context)).sort((a, b) => a - b);
	const medianMs = quantile(sorted, 0.5);

	return {
		...entry,
		medianMs,
		p90Ms: quantile(sorted, 0.9),
		minMs: sorted[0],
		ticksPerSecond: 1000 / medianMs,
		placement,
		offGpuNodes,
		offGpuShapeOps: offGpuOps.filter(op => shapeOps.has(op)),
		offGpuComputeOps: offGpuOps.filter(op => !shapeOps.has(op)),
	};
}
