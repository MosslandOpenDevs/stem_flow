import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

async function source(path) {
  return readFile(new URL(path, root), "utf8");
}

test("upload uses the production API instead of substituting demo stems", async () => {
  const studio = await source("app/stem-flow-studio.tsx");
  assert.match(studio, /xhr\.open\("POST", `\$\{API_BASE\}\/api\/jobs`\)/);
  assert.match(studio, /pollJob\(created\.jobId\)/);
  assert.doesNotMatch(studio, /handleFile[\s\S]{0,200}loadDemo/);
});

test("mixer exposes synchronized transport and real export endpoints", async () => {
  const studio = await source("app/stem-flow-studio.tsx");
  assert.match(studio, /source\.start\(scheduled, safeOffset\)/);
  assert.match(studio, /job\.mixUrl/);
  assert.match(studio, /job\.archiveUrl/);
  assert.match(studio, /toWaveform\(decoded\)/);
});

test("v4 worker preserves residual reconstruction", async () => {
  const pipeline = await source("worker/separate_v4.py");
  assert.match(pipeline, /VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12\.9755\.ckpt"/);
  assert.match(pipeline, /SIX_STEM_MODEL = "BS-Roformer-SW\.ckpt"/);
  assert.match(pipeline, /audio\["other"\] = mix - sum/);
  assert.match(pipeline, /shutil\.make_archive/);
});

test("production worker server-renders the StemFlow studio", async (t) => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  // dist/ is a build artifact and gitignored, so a fresh clone has none.
  if (!existsSync(workerUrl)) {
    t.skip("run `npm run build` first to check the production bundle");
    return;
  }
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const response = await worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /StemFlow/);
  assert.match(html, /Every sound/);
  assert.match(html, /GPU-POWERED STEM SEPARATION/);
});
