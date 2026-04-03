// Seeded so a session can be replayed exactly
export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller: each pair of uniforms gives two standard normals
export function fillNormal(out: Float32Array, random: () => number): void {
  for (let i = 0; i < out.length; i += 2) {
    // 1 - u is in (0, 1], so the log is finite
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    out[i] = radius * Math.cos(angle);
    if (i + 1 < out.length) out[i + 1] = radius * Math.sin(angle);
  }
}
