"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";

type Stem = {
  id: string;
  label: string;
  color: string;
  previewUrl: string;
  downloadUrl: string;
};
type Channel = { volume: number; muted: boolean; solo: boolean };
type JobStatus = {
  jobId: string;
  originalFilename: string;
  status: "uploading" | "queued" | "processing" | "completed" | "failed" | "cancelled";
  progress: number;
  stage: string;
  duration?: number;
  error?: string;
  stems?: Omit<Stem, "color">[];
  archiveUrl?: string;
  mixUrl?: string;
  expiresAt?: string;
};

const COLORS = ["#ff5c7a", "#ff9f43", "#5fe1a5", "#4cc9f0", "#8f7cff", "#cf7cff"];
// Optional local-only sample. Drop six MP3s named after the stems into
// public/demo/ to enable the demo button; the directory is gitignored so no
// audio ships with the repository.
const DEMO_STEMS: Stem[] = [
  { id: "vocals", label: "Vocals", color: COLORS[0], previewUrl: "/demo/Vocals.mp3", downloadUrl: "/demo/Vocals.mp3" },
  { id: "drums", label: "Drums", color: COLORS[1], previewUrl: "/demo/Drums.mp3", downloadUrl: "/demo/Drums.mp3" },
  { id: "bass", label: "Bass", color: COLORS[2], previewUrl: "/demo/Bass.mp3", downloadUrl: "/demo/Bass.mp3" },
  { id: "guitar", label: "Guitar", color: COLORS[3], previewUrl: "/demo/Guitar.mp3", downloadUrl: "/demo/Guitar.mp3" },
  { id: "piano", label: "Piano", color: COLORS[4], previewUrl: "/demo/Piano.mp3", downloadUrl: "/demo/Piano.mp3" },
  { id: "other", label: "Other", color: COLORS[5], previewUrl: "/demo/Other.mp3", downloadUrl: "/demo/Other.mp3" },
];
const API_BASE = (process.env.NEXT_PUBLIC_STEMFLOW_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

function initialChannels(stems: Stem[]) {
  return Object.fromEntries(stems.map((stem) => [stem.id, { volume: 1, muted: false, solo: false }])) as Record<string, Channel>;
}

function formatTime(value: number) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}`;
}

function toWaveform(buffer: AudioBuffer, bars = 72) {
  const data = buffer.getChannelData(0);
  const block = Math.max(1, Math.floor(data.length / bars));
  return Array.from({ length: bars }, (_, index) => {
    let peak = 0;
    const start = index * block;
    const end = Math.min(data.length, start + block);
    for (let cursor = start; cursor < end; cursor += Math.max(1, Math.floor(block / 180))) {
      peak = Math.max(peak, Math.abs(data[cursor]));
    }
    return Math.max(7, Math.min(100, peak * 135));
  });
}

function Waveform({ color, values }: { color: string; values: number[] }) {
  return (
    <div className="meter" aria-hidden="true">
      {(values.length ? values : Array.from({ length: 72 }, () => 8)).map((height, index) => (
        <i key={index} style={{ height: `${height}%`, background: color }} />
      ))}
    </div>
  );
}

async function responseMessage(response: Response) {
  try {
    const payload = await response.json();
    return payload.detail || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

function triggerDownload(url: string, filename?: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  if (filename) anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}

export default function StemFlowStudio() {
  const [stems, setStems] = useState<Stem[]>(DEMO_STEMS);
  const [channels, setChannels] = useState<Record<string, Channel>>(() => initialChannels(DEMO_STEMS));
  const [waveforms, setWaveforms] = useState<Record<string, number[]>>({});
  const [job, setJob] = useState<JobStatus | null>(null);
  const [mode, setMode] = useState<"idle" | "demo" | "job">("idle");
  const [apiOnline, setApiOnline] = useState(false);
  // The demo audio is not part of the repository, so a fresh checkout has none.
  const [demoAvailable, setDemoAvailable] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [loadingAudio, setLoadingAudio] = useState(false);
  const [audioReady, setAudioReady] = useState(false);
  const [error, setError] = useState("");
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [masterVolume, setMasterVolume] = useState(0.9);
  const [dragging, setDragging] = useState(false);
  const [exporting, setExporting] = useState(false);

  const audioContext = useRef<AudioContext | null>(null);
  const buffers = useRef<Map<string, AudioBuffer>>(new Map());
  const sources = useRef<AudioBufferSourceNode[]>([]);
  const gains = useRef<Map<string, GainNode>>(new Map());
  const masterGain = useRef<GainNode | null>(null);
  const startedAt = useRef(0);
  const offset = useRef(0);
  const playingRef = useRef(false);
  const raf = useRef(0);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const ensureContext = useCallback(() => {
    if (!audioContext.current) {
      const AudioCtor = window.AudioContext ||
        (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const context = new AudioCtor();
      const output = context.createGain();
      output.gain.value = masterVolume;
      output.connect(context.destination);
      audioContext.current = context;
      masterGain.current = output;
    }
    return audioContext.current;
  }, [masterVolume]);

  const stopNodes = useCallback(() => {
    sources.current.forEach((source) => {
      try { source.stop(); } catch {}
      source.disconnect();
    });
    gains.current.forEach((node) => node.disconnect());
    sources.current = [];
    gains.current.clear();
    playingRef.current = false;
  }, []);

  const audibleLevel = useCallback((id: string, state = channels) => {
    const anySolo = Object.values(state).some((channel) => channel.solo);
    const channel = state[id];
    if (!channel) return 0;
    return (anySolo ? channel.solo : !channel.muted) ? channel.volume : 0;
  }, [channels]);

  const loadStemAudio = useCallback(async (nextStems: Stem[]) => {
    setLoadingAudio(true);
    setAudioReady(false);
    setError("");
    stopNodes();
    setPlaying(false);
    setPosition(0);
    offset.current = 0;
    const context = ensureContext();
    const nextBuffers = new Map<string, AudioBuffer>();
    const nextWaveforms: Record<string, number[]> = {};
    try {
      await Promise.all(nextStems.map(async (stem) => {
        const response = await fetch(stem.previewUrl);
        if (!response.ok) throw new Error(await responseMessage(response));
        const decoded = await context.decodeAudioData(await response.arrayBuffer());
        nextBuffers.set(stem.id, decoded);
        nextWaveforms[stem.id] = toWaveform(decoded);
      }));
      const durations = [...nextBuffers.values()].map((buffer) => buffer.duration);
      const shortest = Math.min(...durations);
      const longest = Math.max(...durations);
      if (!Number.isFinite(shortest) || longest - shortest > 0.1) {
        throw new Error("Separated stems are not sample-aligned.");
      }
      buffers.current = nextBuffers;
      setDuration(shortest);
      setWaveforms(nextWaveforms);
      setChannels(initialChannels(nextStems));
      setStems(nextStems);
      setAudioReady(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load separated audio.");
      throw reason;
    } finally {
      setLoadingAudio(false);
    }
  }, [ensureContext, stopNodes]);

  const pollJob = useCallback(async (jobId: string) => {
    async function run(): Promise<void> {
      const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
      if (!response.ok) throw new Error(await responseMessage(response));
      const nextJob: JobStatus = await response.json();
      setJob(nextJob);
      if (nextJob.status === "completed" && nextJob.stems) {
        const jobStems = nextJob.stems.map((stem, index) => ({ ...stem, color: COLORS[index] }));
        await loadStemAudio(jobStems);
        setMode("job");
        return;
      }
      if (nextJob.status === "failed" || nextJob.status === "cancelled") {
        setError(nextJob.error || `Job ${nextJob.status}.`);
        return;
      }
      pollTimer.current = setTimeout(() => {
        run().catch((reason) => setError(reason.message));
      }, 1200);
    }
    await run();
  }, [loadStemAudio]);

  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2200);
    fetch(`${API_BASE}/api/health`, { signal: controller.signal })
      .then(async (response) => {
        const health = response.ok ? await response.json() : null;
        setApiOnline(Boolean(health?.gpuWorker));
      })
      .catch(() => setApiOnline(false))
      .finally(() => clearTimeout(timeout));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch(DEMO_STEMS[0].previewUrl, { method: "HEAD", signal: controller.signal })
      .then((response) => setDemoAvailable(response.ok))
      .catch(() => setDemoAvailable(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!playing) return;
    const tick = () => {
      const context = audioContext.current;
      if (!context) return;
      const next = Math.max(0, Math.min(duration, offset.current + context.currentTime - startedAt.current));
      setPosition(next);
      if (next >= duration) {
        stopNodes();
        setPlaying(false);
        offset.current = 0;
        setPosition(0);
        return;
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, duration, stopNodes]);

  useEffect(() => {
    const context = audioContext.current;
    if (!context) return;
    const now = context.currentTime;
    for (const [id, node] of gains.current) {
      node.gain.cancelScheduledValues(now);
      node.gain.setTargetAtTime(audibleLevel(id), now, 0.018);
    }
  }, [channels, audibleLevel]);

  useEffect(() => {
    const context = audioContext.current;
    if (!context || !masterGain.current) return;
    masterGain.current.gain.setTargetAtTime(masterVolume, context.currentTime, 0.018);
  }, [masterVolume]);

  useEffect(() => () => {
    if (pollTimer.current) clearTimeout(pollTimer.current);
    cancelAnimationFrame(raf.current);
    stopNodes();
    audioContext.current?.close();
  }, [stopNodes]);

  const startPlayback = useCallback(async (startOffset: number) => {
    if (!buffers.current.size || duration <= 0) return;
    const context = ensureContext();
    if (context.state === "suspended") await context.resume();
    stopNodes();
    const scheduled = context.currentTime + 0.06;
    const safeOffset = Math.max(0, Math.min(startOffset, duration - 0.02));
    sources.current = stems.map((stem) => {
      const source = context.createBufferSource();
      const gain = context.createGain();
      source.buffer = buffers.current.get(stem.id)!;
      gain.gain.value = audibleLevel(stem.id);
      source.connect(gain).connect(masterGain.current!);
      source.start(scheduled, safeOffset);
      gains.current.set(stem.id, gain);
      return source;
    });
    offset.current = safeOffset;
    startedAt.current = scheduled;
    playingRef.current = true;
    setPlaying(true);
  }, [audibleLevel, duration, ensureContext, stems, stopNodes]);

  async function togglePlay() {
    if (loadingAudio || !buffers.current.size) return;
    if (playingRef.current) {
      offset.current = position;
      stopNodes();
      setPlaying(false);
    } else {
      await startPlayback(offset.current);
    }
  }

  async function seek(value: number) {
    const wasPlaying = playingRef.current;
    stopNodes();
    offset.current = value;
    setPosition(value);
    setPlaying(false);
    if (wasPlaying) await startPlayback(value);
  }

  function patchChannel(id: string, patch: Partial<Channel>) {
    setChannels((previous) => ({
      ...previous,
      [id]: { ...previous[id], ...patch },
    }));
  }

  async function upload(file?: File) {
    if (!file) return;
    if (!apiOnline) {
      setError("The local GPU service is offline. Start it, then retry the upload.");
      return;
    }
    setError("");
    setMode("idle");
    setJob({
      jobId: "",
      originalFilename: file.name,
      status: "uploading",
      progress: 0,
      stage: "Uploading audio",
    });
    setUploadProgress(0);
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/jobs`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) setUploadProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onerror = () => setError("Upload failed. Check the local GPU service.");
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        try {
          setError(JSON.parse(xhr.responseText).detail || "Upload failed.");
        } catch {
          setError(`Upload failed (${xhr.status}).`);
        }
        return;
      }
      const created: JobStatus = JSON.parse(xhr.responseText);
      setJob(created);
      pollJob(created.jobId).catch((reason) => setError(reason.message));
    };
    xhr.send(form);
  }

  async function loadDemo() {
    setJob(null);
    setMode("demo");
    await loadStemAudio(DEMO_STEMS);
  }

  async function downloadMix() {
    if (mode !== "job" || !job?.mixUrl) return;
    setExporting(true);
    setError("");
    try {
      const levels = Object.fromEntries(stems.map((stem) => [stem.id, audibleLevel(stem.id)]));
      const response = await fetch(job.mixUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ levels }),
      });
      if (!response.ok) throw new Error(await responseMessage(response));
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      triggerDownload(url, `${job.originalFilename.replace(/\.[^.]+$/, "")}-mix.wav`);
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Mix export failed.");
    } finally {
      setExporting(false);
    }
  }

  async function removeJob() {
    if (!job?.jobId) return;
    const response = await fetch(`${API_BASE}/api/jobs/${job.jobId}`, { method: "DELETE" });
    if (!response.ok && response.status !== 202) {
      setError(await responseMessage(response));
      return;
    }
    if (pollTimer.current) clearTimeout(pollTimer.current);
    stopNodes();
    setPlaying(false);
    setJob(null);
    setMode("idle");
    setPosition(0);
    setDuration(0);
    setAudioReady(false);
    buffers.current.clear();
  }

  const processing = job && !["completed", "failed", "cancelled"].includes(job.status);
  const ready = audioReady && duration > 0 && !loadingAudio;
  const sessionName = job?.originalFilename || (mode === "demo" ? "Demo track" : "No track loaded");

  return (
    <main>
      <header>
        <a className="brand" href="#" aria-label="StemFlow home"><span>◒</span> StemFlow</a>
        <nav><a href="#studio">Studio</a><a href="#how">How it works</a><a href="#about">About</a></nav>
        <div className={`service-state ${apiOnline ? "online" : "offline"}`}>
          <i /> {apiOnline ? "GPU service online" : "Demo mode"}
        </div>
      </header>

      <section className="hero" id="studio">
        <div className="hero-copy">
          <span className="eyebrow"><b>●</b> GPU-POWERED STEM SEPARATION</span>
          <h1>Every sound,<br /><em>in your hands.</em></h1>
          <p>Upload a track, separate six sample-aligned stems with the v4 cascade, then shape and export a mix that is entirely yours.</p>
          <div className="trust"><span>6</span> precision stems <i /> <span>32-bit float</span> masters <i /> <span>24-hour</span> automatic expiry</div>
        </div>
        <div
          className={`upload-card ${dragging ? "is-dragging" : ""}`}
          onDragOver={(event: DragEvent) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event: DragEvent) => {
            event.preventDefault();
            setDragging(false);
            upload(event.dataTransfer.files[0]);
          }}
        >
          <div className="upload-icon">{processing ? "⌁" : "↑"}</div>
          <h2>{job?.originalFilename || "Drop your audio here"}</h2>
          <p>
            {processing
              ? job.stage
              : apiOnline
                ? "MP3, WAV, FLAC, M4A, AAC or OGG · up to 500 MB"
                : "Start the local GPU service to process new audio"}
          </p>
          {processing && (
            <div className="progress" role="progressbar" aria-valuenow={job.status === "uploading" ? uploadProgress : job.progress}>
              <i style={{ width: `${job.status === "uploading" ? uploadProgress : job.progress}%` }} />
              <span>{job.status === "uploading" ? uploadProgress : job.progress}%</span>
            </div>
          )}
          {!processing && (
            <label className={`primary ${!apiOnline ? "disabled" : ""}`}>
              Choose audio
              <input
                type="file"
                accept=".mp3,.wav,.flac,.m4a,.aac,.ogg,audio/*"
                disabled={!apiOnline}
                onChange={(event: ChangeEvent<HTMLInputElement>) => upload(event.target.files?.[0])}
              />
            </label>
          )}
          {!processing && demoAvailable && <button className="demo-link" onClick={loadDemo}>Explore with the demo track →</button>}
          {processing && <button className="demo-link danger-link" onClick={removeJob}>Cancel processing</button>}
          {error && <p className="inline-error" role="alert">{error}</p>}
        </div>
      </section>

      <section className={`studio ${ready ? "is-ready" : ""}`}>
        <div className="studio-title">
          <div>
            <span className="live-dot" />
            <small>{mode === "job" ? "YOUR SESSION" : mode === "demo" ? "DEMO SESSION" : "STUDIO"}</small>
            <h2>{sessionName}</h2>
          </div>
          <div className="session-actions">
            {job?.expiresAt && <span className="expiry">Temporary files expire automatically</span>}
            <div className="status-pill">{loadingAudio ? "Decoding stems" : ready ? "Ready to mix" : "Waiting for audio"}</div>
            {mode === "job" && <button className="icon-button" onClick={removeJob} title="Delete session">×</button>}
          </div>
        </div>

        <div className="transport">
          <button className="play" onClick={togglePlay} disabled={!ready} aria-label={playing ? "Pause" : "Play"}>{playing ? "Ⅱ" : "▶"}</button>
          <span>{formatTime(position)}</span>
          <input aria-label="Playback position" type="range" min="0" max={duration || 1} step=".01" value={position} onChange={(event) => seek(Number(event.target.value))} disabled={!ready} />
          <span>{formatTime(duration)}</span>
          <button className="reset" onClick={() => seek(0)} disabled={!ready} aria-label="Return to start">↶</button>
          <label className="master-control">MASTER
            <input aria-label="Master volume" type="range" min="0" max="1" step=".01" value={masterVolume} onChange={(event) => setMasterVolume(Number(event.target.value))} />
          </label>
        </div>

        <div className="channels">
          {stems.map((stem, index) => {
            const channel = channels[stem.id] || { volume: 1, muted: false, solo: false };
            return (
              <article className="channel" key={stem.id} style={{ "--accent": stem.color } as React.CSSProperties}>
                <div className="channel-head">
                  <span className="stem-icon">{["◉", "✦", "≋", "⌁", "▥", "◇"][index]}</span>
                  <strong>{stem.label}</strong>
                  <button onClick={() => triggerDownload(stem.downloadUrl, `${stem.label}.${mode === "job" ? "wav" : "mp3"}`)} disabled={!ready} title={`Download ${stem.label}`}>↓</button>
                </div>
                <Waveform color={stem.color} values={waveforms[stem.id] || []} />
                <div className="channel-controls">
                  <button className={channel.muted ? "mute active" : "mute"} onClick={() => patchChannel(stem.id, { muted: !channel.muted })} aria-pressed={channel.muted}>M</button>
                  <button className={channel.solo ? "solo active" : "solo"} onClick={() => patchChannel(stem.id, { solo: !channel.solo })} aria-pressed={channel.solo}>S</button>
                  <input aria-label={`${stem.label} volume`} type="range" min="0" max="1" step=".01" value={channel.volume} onChange={(event) => patchChannel(stem.id, { volume: Number(event.target.value) })} />
                  <output>{Math.round(channel.volume * 100)}%</output>
                </div>
              </article>
            );
          })}
        </div>

        <div className="export-bar">
          <div><strong>Ready to take it with you?</strong><span>{mode === "job" ? "Export this balance or download all lossless stems." : "Process your own track to unlock lossless exports."}</span></div>
          <button className="ghost" onClick={downloadMix} disabled={mode !== "job" || !ready || exporting}>{exporting ? "Rendering…" : "↓ Current mix WAV"}</button>
          <button className="primary" onClick={() => job?.archiveUrl && triggerDownload(job.archiveUrl)} disabled={mode !== "job" || !ready}>Download stems ZIP</button>
        </div>
      </section>

      <section className="how" id="how">
        <p className="eyebrow">FROM TRACK TO FULL CONTROL</p>
        <h2>Three steps. One aligned timeline.</h2>
        <div>
          <article><b>01</b><span>↑</span><h3>Upload</h3><p>Your file is stored under an isolated job ID and expires automatically.</p></article>
          <article><b>02</b><span>⌘</span><h3>Separate</h3><p>Two GPU models isolate vocals and five instrumental layers.</p></article>
          <article><b>03</b><span>≋</span><h3>Shape & export</h3><p>Mix sample-aligned stems and export 24-bit WAV or the lossless pack.</p></article>
        </div>
      </section>
      <footer id="about"><a className="brand" href="#"><span>◒</span> StemFlow</a><p>Upload audio, separate every stem, and create your own mix.</p><small>V4 · BS-Roformer cascade · residual reconstruction</small></footer>
    </main>
  );
}
