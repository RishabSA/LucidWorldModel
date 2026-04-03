import { useEffect, useState } from "react";

import { benchEntry, type Entry, type Result } from "./benchmark.ts";
import { fetchOk, getAdapter, ort, webGpuRequired } from "./runtime.ts";

interface Header {
	vendor: string;
	architecture: string;
	shaderF16: boolean;
	userAgent: string;
	onnxruntimeWeb: string;
}

const modelDir = "/models/bench";
const numberCell = "px-3 py-1 text-right";
const textCell = "px-3 py-1 text-left";
const errorText = "text-red-500 dark:text-red-400";

// StrictMode runs effects twice in dev, and two runs at once would share the GPU and skew the timings
let started = false;

function message(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function Bench() {
	const [header, setHeader] = useState<Header | null>(null);
	const [results, setResults] = useState<Result[]>([]);
	const [status, setStatus] = useState("Checking WebGPU");
	const [error, setError] = useState("");
	const [report, setReport] = useState("");
	const [copied, setCopied] = useState(false);

	useEffect(() => {
		if (started) return;
		started = true;

		const run = async () => {
			const adapter = await getAdapter();
			if (!adapter) throw new Error(webGpuRequired);
			// adapter.info is missing before Chrome 127
			const info = {
				vendor: adapter.info?.vendor ?? "unknown",
				architecture: adapter.info?.architecture ?? "unknown",
				shaderF16: adapter.features.has("shader-f16"),
				userAgent: navigator.userAgent,
				onnxruntimeWeb: ort.env.versions.web ?? "unknown",
			};
			setHeader(info);

			setStatus("Fetching the manifest");
			const entries: Entry[] = await (
				await fetchOk(`${modelDir}/manifest.json`)
			).json();

			const done: Result[] = [];
			for (const [i, entry] of entries.entries()) {
				setStatus(`Running ${entry.name} (${i + 1} of ${entries.length})`);
				try {
					const bytes = new Uint8Array(
						await (await fetchOk(`${modelDir}/${entry.file}`)).arrayBuffer(),
					);
					done.push(await benchEntry(entry, bytes));
				} catch (entryError) {
					done.push({ ...entry, error: message(entryError) });
				}
				setResults([...done]);
			}

			const json = JSON.stringify({ header: info, results: done });
			console.log(`BENCH_RESULTS ${json}`);
			setReport(json);
			setStatus("Done");
		};

		run().catch((runError: unknown) => {
			setError(message(runError));
			setStatus("Stopped");
		});
	}, []);

	const copy = async () => {
		try {
			await navigator.clipboard.writeText(report);
			setCopied(true);
		} catch (copyError) {
			setError(`Copy failed: ${message(copyError)}`);
		}
	};

	return (
		<main className="space-y-4 p-8">
			<h1 className="text-3xl font-bold">Browser latency benchmark</h1>
			{header && (
				<p className="text-neutral-700 dark:text-neutral-300">
					GPU: {header.vendor} / {header.architecture}, shader-f16:{" "}
					{header.shaderF16 ? "yes" : "no"}, onnxruntime-web{" "}
					{header.onnxruntimeWeb}
					<br />
					{header.userAgent}
				</p>
			)}
			{error && <p className={errorText}>{error}</p>}
			<p className="flex items-center gap-3">
				{status}
				{report && (
					<button
						type="button"
						onClick={copy}
						className="cursor-pointer rounded bg-blue-600 px-3 py-1 text-sm text-white transition-colors hover:bg-blue-700 dark:bg-blue-500 dark:text-white dark:hover:bg-blue-600">
						{copied ? "Copied" : "Copy results"}
					</button>
				)}
			</p>
			{results.length > 0 && (
				<table className="border-collapse tabular-nums">
					<thead>
						<tr className="border-b border-neutral-300 font-semibold dark:border-neutral-600">
							<th className={textCell}>Name</th>
							<th className={numberCell}>Width</th>
							<th className={numberCell}>K</th>
							<th className={numberCell}>Params</th>
							<th className={numberCell}>GFLOPs</th>
							<th className={numberCell}>Median ms</th>
							<th className={numberCell}>p90 ms</th>
							<th className={numberCell}>Min ms</th>
							<th className={numberCell}>Max ticks/s</th>
							<th className={textCell}>Off WebGPU</th>
						</tr>
					</thead>
					<tbody>
						{results.map(result => (
							<tr
								key={result.name}
								className="border-b border-neutral-200 dark:border-neutral-700">
								<td className={textCell}>{result.name}</td>
								<td className={numberCell}>{result.width}</td>
								<td className={numberCell}>{result.steps}</td>
								<td className={numberCell}>
									{(result.params / 1e6).toFixed(2)}M
								</td>
								<td className={numberCell}>{result.gflops.toFixed(2)}</td>
								{result.error ? (
									<td colSpan={5} className={`${textCell} ${errorText}`}>
										{result.error}
									</td>
								) : (
									<>
										<td className={numberCell}>
											{result.medianMs?.toFixed(2)}
										</td>
										<td className={numberCell}>{result.p90Ms?.toFixed(2)}</td>
										<td className={numberCell}>{result.minMs?.toFixed(2)}</td>
										<td className={numberCell}>
											{result.ticksPerSecond?.toFixed(0)}
										</td>
										<td className={textCell}>
											{result.offGpuNodes?.length === 0 && "none"}
											{result.offGpuShapeOps?.join(", ")}{" "}
											{result.offGpuComputeOps?.length ? (
												<strong className={errorText}>
													compute: {result.offGpuComputeOps.join(", ")}
												</strong>
											) : null}
										</td>
									</>
								)}
							</tr>
						))}
					</tbody>
				</table>
			)}
		</main>
	);
}

export default Bench;
