# Tauri-Transcribe
Tauri-based cross-platform transcription app using transformers.js with WebGPU acceleration

## Features
- Audio/Video to text transcription
- Uses Unravler/parakeet-tdt-0.6b-v2-onnx model via transformers.js v4
- WebGPU acceleration for fast inference
- Model caching between runs
- Clean dark theme interface
- Copy to clipboard and download as .txt

## Supported Formats
- Audio: MP3, WAV, OGG, FLAC, M4A, AAC
- Video: MP4, WEBM, MOV, AVI, MKV

## Quick Start
```bash
# Install dependencies
npm install

# Development
npm run dev

# Build frontend
npm run build
```

## Desktop Build (requires Rust and system dependencies)
```bash
npm run tauri build
```

## Requirements
- Node.js 18+
- For desktop builds: Rust 1.70+ and system dependencies (see SPEC.md)
