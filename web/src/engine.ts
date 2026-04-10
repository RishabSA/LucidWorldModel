import { fillNormal, mulberry32 } from "./noise.ts";
import { message, ort } from "./runtime.ts";

export interface GameEntry {
	game: string;
	title: string;
	meta: string;
}

// One exported graph per sampler step count
export interface Graph {
	steps: number;
	file: string;
	bytes: number;
}

export interface Meta {
	// The default graph, also models[0]
	model: string;
	models: Graph[];
	tick_hz: number;
	context: number;
	input_mode: "queue" | "hold";
	keys: Record<string, number>;
	sampler: { steps: number };
	starts: { file: string; count: number };
	parity: string;
}

export const modelDir = "/models";
export const frameSize = 64;
export const pixels = frameSize * frameSize;
const warmupRuns = 3;

// onnxruntime-web's WebGPU build cannot overlap wasm calls, even across sessions; overlaps fail in BufferManager::Download or out of bounds
let pending: Promise<unknown> = Promise.resolve();

function exclusive<T>(task: () => Promise<T>): Promise<T> {
	const result = pending.then(task);
	// The chain only orders the calls; each caller still gets its own error
	pending = result.catch(() => undefined);
	return result;
}

// CHW floats in [-1, 1], already palette-snapped, to RGBA bytes
export function toRgba(frame: Float32Array): Uint8ClampedArray<ArrayBuffer> {
	const rgba = new Uint8ClampedArray(pixels * 4);

	for (let pixel = 0; pixel < pixels; pixel++) {
		for (let channel = 0; channel < 3; channel++) {
			rgba[pixel * 4 + channel] = Math.round(
				(frame[channel * pixels + pixel] + 1) * 127.5,
			);
		}
		rgba[pixel * 4 + 3] = 255;
	}

	return rgba;
}

// Calls must not overlap: each one reads and writes the shared buffers
export class Engine {
	// shape: (3N, 64, 64), oldest frame first, in [-1, 1]
	readonly context: Float32Array;
	readonly actions: BigInt64Array;
	// shape: (3(N + 1), 64, 64)
	readonly noise: Float32Array;
	readonly n: number;
	session: ort.InferenceSession | null = null;
	feeds: Record<string, ort.Tensor>;
	random = mulberry32(0);
	// Time of the last step's session.run, including the readback
	runMs = 0;

	constructor(n: number) {
		this.n = n;
		this.context = new Float32Array(3 * n * pixels);
		this.actions = new BigInt64Array(n);
		this.noise = new Float32Array(3 * (n + 1) * pixels);
		// The tensors wrap the buffers, so every run reads their current contents
		this.feeds = {
			context: new ort.Tensor("float32", this.context, [
				1,
				3 * n,
				frameSize,
				frameSize,
			]),
			actions: new ort.Tensor("int64", this.actions, [1, n]),
			noise: new ort.Tensor("float32", this.noise, [
				1,
				3 * (n + 1),
				frameSize,
				frameSize,
			]),
		};
	}

	// Keeps the context, actions and noise generator, so a game in progress goes on with the new graph
	async setModel(model: Uint8Array): Promise<void> {
		await this.release();

		try {
			this.session = await exclusive(() =>
				ort.InferenceSession.create(model, {
					executionProviders: ["webgpu"],
				}),
			);
			// The first run compiles shaders and is slow
			for (let i = 0; i < warmupRuns; i++) await this.run();
		} catch (createError) {
			throw new Error(
				`Creating the ONNX session failed: ${message(createError)}`,
			);
		}
	}

	// The N - 1 start actions fill the end, so step() shifts the first slot out
	load(frames: Uint8Array, actions: ArrayLike<number>): void {
		for (let i = 0; i < this.n * pixels; i++) {
			const frame = Math.floor(i / pixels);
			for (let channel = 0; channel < 3; channel++) {
				this.context[(frame * 3 + channel) * pixels + (i % pixels)] =
					frames[i * 3 + channel] / 127.5 - 1;
			}
		}

		this.actions.fill(0n);
		for (let i = 0; i < actions.length; i++) {
			this.actions[this.n - actions.length + i] = BigInt(actions[i]);
		}
	}

	// Starts file: uint8 frames [count, N, 64, 64, 3], then uint8 actions [count, N - 1]
	restart(starts: Uint8Array, count: number, seed: number): Float32Array {
		const frameBytes = this.n * pixels * 3;
		const size = count * (frameBytes + this.n - 1);
		if (starts.length !== size) {
			throw new Error(
				`The starts file has ${starts.length} bytes, but ${count} starts with context ${this.n} need ${size}`,
			);
		}

		this.random = mulberry32(seed);
		// Drawn from the seeded generator, so the seed and the keys pressed replay a session exactly
		const index = Math.floor(this.random() * count);
		const actionsStart = count * frameBytes + index * (this.n - 1);
		this.load(
			starts.subarray(index * frameBytes, (index + 1) * frameBytes),
			starts.subarray(actionsStart, actionsStart + this.n - 1),
		);

		return this.context.subarray(3 * (this.n - 1) * pixels);
	}

	run(): Promise<Float32Array> {
		return exclusive(async () => {
			const { frame } = await this.session!.run(this.feeds);
			return (await frame.getData()) as Float32Array;
		});
	}

	async step(action: number): Promise<Float32Array> {
		this.actions.copyWithin(0, 1);
		this.actions[this.n - 1] = BigInt(action);
		fillNormal(this.noise, this.random);

		const start = performance.now();
		const frame = await this.run();
		this.runMs = performance.now() - start;
		this.context.copyWithin(0, 3 * pixels);
		this.context.set(frame, 3 * (this.n - 1) * pixels);

		return frame;
	}

	async release(): Promise<void> {
		const session = this.session;
		this.session = null;
		if (session) await exclusive(() => session.release());
	}
}
