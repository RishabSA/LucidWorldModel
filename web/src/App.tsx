import { useEffect, useRef, useState } from "react";

import {
	Engine,
	frameSize,
	modelDir,
	toRgba,
	type GameEntry,
	type Meta,
} from "./engine.ts";
import { listen } from "./input.ts";
import {
	fetchBytes,
	fetchOk,
	getAdapter,
	message,
	webGpuRequired,
} from "./runtime.ts";

interface Stats {
	runMs: number;
	tickRate: number;
	ticks: number;
	seed: number;
}

// Ticks behind the median run time and the effective tick rate
const statWindow = 50;
const pausePollMs = 50;
// Below this fraction of tick_hz, the tick rate shows as a warning
const slowFraction = 0.95;
const button =
	"cursor-pointer rounded-md border border-neutral-300 px-4 py-1.5 text-sm transition-colors hover:bg-neutral-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-neutral-700 dark:hover:bg-neutral-800";
const label = "text-neutral-500 dark:text-neutral-400";

function sleep(ms: number): Promise<void> {
	return new Promise(resolve => setTimeout(resolve, ms));
}

function median(values: number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	const middle = Math.floor(sorted.length / 2);
	return sorted.length % 2
		? sorted[middle]
		: (sorted[middle - 1] + sorted[middle]) / 2;
}

function App() {
	const [games, setGames] = useState<GameEntry[]>([]);
	const [entry, setEntry] = useState<GameEntry | null>(null);
	const [meta, setMeta] = useState<Meta | null>(null);
	const [stats, setStats] = useState<Stats | null>(null);
	const [loading, setLoading] = useState("Checking WebGPU");
	const [progress, setProgress] = useState(0);
	const [error, setError] = useState("");
	const [paused, setPaused] = useState(false);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	// The tick loop reads these between ticks, so a click never interrupts a session.run
	const pausedRef = useRef<boolean>(false);
	const restartRef = useRef<boolean>(false);

	useEffect(() => {
		let cancelled = false;

		const start = async () => {
			if (!(await getAdapter())) throw new Error(webGpuRequired);
			const manifest: GameEntry[] = await (
				await fetchOk(`${modelDir}/manifest.json`)
			).json();
			if (cancelled) return;
			setGames(manifest);
			setEntry(manifest[0]);
		};

		start().catch((startError: unknown) => setError(message(startError)));
		return () => {
			cancelled = true;
		};
	}, []);

	useEffect(() => {
		if (!entry) return;
		let cancelled = false;

		const play = async () => {
			setMeta(null);
			setStats(null);
			setLoading(`Loading ${entry.title}`);
			setProgress(0);
			const meta: Meta = await (
				await fetchOk(`${modelDir}/${entry.meta}`)
			).json();
			const download = async (file: string) => {
				setLoading(`Downloading ${file}`);
				setProgress(0);
				return fetchBytes(`${modelDir}/${file}`, setProgress);
			};
			const model = await download(meta.model);
			const starts = await download(meta.starts.file);
			if (cancelled) return;

			setLoading("Compiling shaders");
			let engine: Engine;
			try {
				engine = await Engine.create(model, meta.context);
			} catch (createError) {
				throw new Error(
					`Creating the ONNX session failed: ${message(createError)}`,
				);
			}
			const input = listen(meta.keys, meta.input_mode);

			try {
				const period = 1000 / meta.tick_hz;
				let seed = 0;
				let ticks = 0;
				const runMs: number[] = [];
				const tickStarts: number[] = [];
				const draw = (frame: Float32Array) => {
					canvasRef.current
						?.getContext("2d")
						?.putImageData(
							new ImageData(toRgba(frame), frameSize, frameSize),
							0,
							0,
						);
				};
				const restart = () => {
					seed = Math.floor(Math.random() * 2 ** 32);
					ticks = 0;
					runMs.length = 0;
					tickStarts.length = 0;
					input.clear();
					draw(engine.restart(starts, meta.starts.count, seed));
					setStats({ runMs: 0, tickRate: 0, ticks, seed });
					setMeta(meta);
					setLoading("");
				};

				restartRef.current = true;
				while (!cancelled) {
					if (restartRef.current) {
						restartRef.current = false;
						restart();
					}
					if (pausedRef.current) {
						// Keys pressed while paused are dropped, and the tick rate restarts after the pause
						input.clear();
						tickStarts.length = 0;
						await sleep(pausePollMs);
						continue;
					}

					const start = performance.now();
					const frame = await engine.step(input.next());
					if (cancelled) break;
					runMs.push(engine.runMs);
					tickStarts.push(start);
					if (runMs.length > statWindow) runMs.shift();
					if (tickStarts.length > statWindow + 1) tickStarts.shift();
					ticks++;
					draw(frame);
					setStats({
						runMs: median(runMs),
						tickRate:
							tickStarts.length > 1
								? ((tickStarts.length - 1) * 1000) /
									(start - tickStarts[0])
								: 0,
						ticks,
						seed,
					});

					await sleep(Math.max(0, period - (performance.now() - start)));
				}
			} finally {
				input.stop();
				await engine.release();
			}
		};

		play().catch((playError: unknown) => {
			if (!cancelled) setError(message(playError));
		});
		return () => {
			cancelled = true;
		};
	}, [entry]);

	useEffect(() => {
		const pause = () => {
			pausedRef.current = true;
			setPaused(true);
		};
		const onVisibilityChange = () => {
			if (document.hidden) pause();
		};

		window.addEventListener("blur", pause);
		document.addEventListener("visibilitychange", onVisibilityChange);
		return () => {
			window.removeEventListener("blur", pause);
			document.removeEventListener("visibilitychange", onVisibilityChange);
		};
	}, []);

	const pick = (game: GameEntry) => {
		pausedRef.current = false;
		setPaused(false);
		setEntry(game);
	};

	const restart = () => {
		restartRef.current = true;
		pausedRef.current = false;
		setPaused(false);
	};

	const togglePause = () => {
		pausedRef.current = !pausedRef.current;
		setPaused(pausedRef.current);
	};

	const slow =
		meta !== null &&
		stats !== null &&
		stats.tickRate > 0 &&
		stats.tickRate < slowFraction * meta.tick_hz;

	return (
		<main className="mx-auto max-w-4xl space-y-8 px-4 py-10 md:px-8">
			<header className="space-y-2">
				<h1 className="text-3xl font-semibold tracking-tight">
					LucidWorldModel
				</h1>
				<p className="text-neutral-600 dark:text-neutral-400">
					Every frame is generated by a model trained on the game. The browser
					runs no game code.
				</p>
			</header>
			{error ? (
				<p className="rounded-md border border-red-500 p-4 text-red-500 dark:border-red-400 dark:text-red-400">
					{error}
				</p>
			) : (
				<div className="flex flex-col gap-8 md:flex-row">
					<div className="relative aspect-square w-full max-w-[512px] shrink-0 overflow-hidden rounded-md bg-neutral-900 dark:bg-neutral-800">
						<canvas
							ref={canvasRef}
							width={frameSize}
							height={frameSize}
							className={`size-full transition-opacity [image-rendering:pixelated] ${paused ? "opacity-40" : ""}`}
						/>
						{loading ? (
							<div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-8 text-sm text-neutral-100 dark:text-neutral-200">
								<p>{loading}</p>
								<div className="h-1.5 w-full max-w-64 overflow-hidden rounded-full bg-neutral-700 dark:bg-neutral-600">
									<div
										className="h-full bg-blue-600 transition-all dark:bg-blue-500"
										style={{ width: `${progress * 100}%` }}
									/>
								</div>
							</div>
						) : (
							paused && (
								<p className="absolute inset-0 flex items-center justify-center text-lg font-semibold text-neutral-100 dark:text-neutral-200">
									Paused
								</p>
							)
						)}
					</div>
					<aside className="flex-1 space-y-6">
						<div className="flex flex-wrap gap-2">
							{games.map(game => (
								<button
									key={game.game}
									type="button"
									aria-pressed={game === entry}
									onClick={() => pick(game)}
									className={`${button} aria-pressed:border-blue-600 aria-pressed:bg-blue-600 aria-pressed:text-white dark:aria-pressed:border-blue-500 dark:aria-pressed:bg-blue-500 dark:aria-pressed:text-white`}>
									{game.title}
								</button>
							))}
						</div>
						<div className="flex gap-2">
							<button
								type="button"
								onClick={restart}
								disabled={loading !== ""}
								className={button}>
								Restart
							</button>
							<button
								type="button"
								onClick={togglePause}
								disabled={loading !== ""}
								className={button}>
								{paused ? "Resume" : "Pause"}
							</button>
						</div>
						{meta && stats && (
							<dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-sm tabular-nums">
								<dt className={label}>Inference ms</dt>
								<dd>{stats.runMs ? stats.runMs.toFixed(1) : "-"}</dd>
								<dt className={label}>Tick rate</dt>
								<dd
									className={
										slow ? "text-yellow-500 dark:text-yellow-400" : ""
									}>
									{stats.tickRate ? stats.tickRate.toFixed(1) : "-"} of{" "}
									{meta.tick_hz} Hz
								</dd>
								<dt className={label}>Sampler steps</dt>
								<dd>{meta.sampler.steps}</dd>
								<dt className={label}>Seed</dt>
								<dd>{stats.seed}</dd>
								<dt className={label}>Ticks</dt>
								<dd>{stats.ticks}</dd>
							</dl>
						)}
					</aside>
				</div>
			)}
		</main>
	);
}

export default App;
