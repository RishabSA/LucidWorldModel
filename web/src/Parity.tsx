import { useEffect, useState } from "react";

import {
	Engine,
	modelDir,
	pixels,
	toRgba,
	type GameEntry,
	type Meta,
} from "./engine.ts";
import { fetchOk, getAdapter, message, webGpuRequired } from "./runtime.ts";

interface ParityCase {
	// base64 uint8 [N, 64, 64, 3]
	frames: string;
	actions: number[];
	// base64 float32 little-endian [3(N + 1), 64, 64]
	noise: string;
	// base64 uint8 [64, 64, 3]
	expected: string;
}

interface ParityFile {
	min_identical: number;
	cases: ParityCase[];
}

interface CaseResult {
	game: string;
	index: number;
	identical: number;
	pass: boolean;
}

const cell = "px-3 py-1 text-left";

// StrictMode runs effects twice in dev, and one check at a time is enough
let started = false;

function decode(base64: string): Uint8Array {
	return Uint8Array.from(atob(base64), char => char.charCodeAt(0));
}

async function fetchJson<T>(url: string): Promise<T> {
	return (await fetchOk(url)).json();
}

function Parity() {
	const [results, setResults] = useState<CaseResult[]>([]);
	const [status, setStatus] = useState("Checking WebGPU");
	const [error, setError] = useState("");

	useEffect(() => {
		if (started) return;
		started = true;

		const run = async () => {
			if (!(await getAdapter())) throw new Error(webGpuRequired);
			const games = await fetchJson<GameEntry[]>(`${modelDir}/manifest.json`);

			const done: CaseResult[] = [];
			for (const entry of games) {
				setStatus(`Checking ${entry.title}`);
				const meta = await fetchJson<Meta>(`${modelDir}/${entry.meta}`);
				const parity = await fetchJson<ParityFile>(
					`${modelDir}/${meta.parity}`,
				);
				const model = new Uint8Array(
					await (await fetchOk(`${modelDir}/${meta.model}`)).arrayBuffer(),
				);
				const engine = await Engine.create(model, meta.context);

				try {
					for (const [index, item] of parity.cases.entries()) {
						engine.load(decode(item.frames), item.actions);
						engine.noise.set(new Float32Array(decode(item.noise).buffer));
						const rgba = toRgba(await engine.run());
						const expected = decode(item.expected);

						let same = 0;
						for (let pixel = 0; pixel < pixels; pixel++) {
							if (
								rgba[pixel * 4] === expected[pixel * 3] &&
								rgba[pixel * 4 + 1] === expected[pixel * 3 + 1] &&
								rgba[pixel * 4 + 2] === expected[pixel * 3 + 2]
							)
								same++;
						}

						const identical = same / pixels;
						done.push({
							game: entry.title,
							index,
							identical,
							pass: identical >= parity.min_identical,
						});
						setResults([...done]);
					}
				} finally {
					await engine.release();
				}
			}

			const passed = done.filter(result => result.pass).length;
			setStatus(`Done: ${passed} of ${done.length} cases pass`);
		};

		run().catch((runError: unknown) => {
			setError(message(runError));
			setStatus("Stopped");
		});
	}, []);

	return (
		<main className="space-y-4 p-8">
			<h1 className="text-3xl font-bold">Parity check</h1>
			<p className="text-neutral-700 dark:text-neutral-300">
				Fixed inputs from the export script, run in the browser and compared
				with the PyTorch output.
			</p>
			{error && <p className="text-red-500 dark:text-red-400">{error}</p>}
			<p>{status}</p>
			{results.length > 0 && (
				<table className="border-collapse tabular-nums">
					<thead>
						<tr className="border-b border-neutral-300 font-semibold dark:border-neutral-600">
							<th className={cell}>Game</th>
							<th className={cell}>Case</th>
							<th className={cell}>Identical pixels</th>
							<th className={cell}>Result</th>
						</tr>
					</thead>
					<tbody>
						{results.map(result => (
							<tr
								key={`${result.game} ${result.index}`}
								className="border-b border-neutral-200 dark:border-neutral-700">
								<td className={cell}>{result.game}</td>
								<td className={cell}>{result.index}</td>
								<td className={cell}>
									{(result.identical * 100).toFixed(3)}%
								</td>
								<td
									className={`${cell} font-semibold ${result.pass ? "text-green-500 dark:text-green-400" : "text-red-500 dark:text-red-400"}`}>
									{result.pass ? "PASS" : "FAIL"}
								</td>
							</tr>
						))}
					</tbody>
				</table>
			)}
		</main>
	);
}

export default Parity;
