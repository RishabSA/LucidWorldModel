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
  if (!response.ok) throw new Error(`Fetching ${url} failed: HTTP ${response.status} ${response.statusText}`);
  return response;
}
