# StemFlow v4 worker

The worker is started by the FastAPI job manager; it is not a separate public
service. `separate_v4.py` invokes the CUDA-enabled `audio-separator`
environment, writes six aligned float WAV stems, generates MP3 previews, and
packages a ZIP.

Use the project-level setup and start commands:

```powershell
.\scripts\setup-local.ps1
.\scripts\start-api.ps1
```

Tool and model locations are resolved in this order: the `STEMFLOW_*`
environment variables (normally set in `.env`), then anything vendored into the
project under `models/`, `vendor/ffmpeg/`, or `.sepenv/`, then `PATH`. See
[Asset resolution](../README.md#asset-resolution) for details.
