import * as ort from "onnxruntime-web/webgpu";

export { ort };

export const webGpuRequired =
	"WebGPU required. Use desktop Chrome or Edge 113+, Safari 26, or Firefox 141+ on Windows.";

// Null when the browser has no usable WebGPU adapter
export async function getAdapter(): Promise<GPUAdapter | null> {
	return (await navigator.gpu?.requestAdapter()) ?? null;
}

export async function fetchOk(url: string): Promise<Response> {
	const response = await fetch(url);
	if (!response.ok)
		throw new Error(
			`Fetching ${url} failed: HTTP ${response.status} ${response.statusText}`,
		);
	return response;
}

// Reports the fraction received, from content-length (no reports when the server omits it)
export async function fetchBytes(
	url: string,
	onProgress: (fraction: number) => void,
): Promise<Uint8Array> {
	const response = await fetchOk(url);
	const total = Number(response.headers.get("content-length"));
	let loaded = 0;
	const counted = response.body!.pipeThrough(
		new TransformStream<Uint8Array, Uint8Array>({
			transform(chunk, controller) {
				loaded += chunk.length;
				if (total) onProgress(Math.min(1, loaded / total));
				controller.enqueue(chunk);
			},
		}),
	);
	return new Uint8Array(await new Response(counted).arrayBuffer());
}

export function message(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}
