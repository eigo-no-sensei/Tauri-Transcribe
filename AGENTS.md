# Tauri-Transcribe Agent Context

## Project Structure
- `src/App.tsx` - Main React component with transcription logic
- `src/main.tsx` - React entry point
- `src/styles.css` - Global CSS styles with dark theme
- `src-tauri/` - Tauri desktop wrapper
- `dist/` - Built frontend output

## Key Technologies
- Tauri 2.x (desktop framework)
- React 18 + TypeScript (frontend)
- @huggingface/transformers v4 (ML inference)
- Vite (build tool)

## Important Notes
- The app uses WebGPU for ML inference - requires browser with WebGPU support
- Model (parakeet-v2-0.6B) is cached in browser IndexedDB via transformers.js
- The ASR pipeline uses the 'automatic-speech-recognition' task with 'webgpu' device
- For desktop builds, Linux requires: pkg-config, libglib2.0-dev, libgtk-3-dev, libsoup-3.0-dev, libwebkit2gtk-4.1-dev

## Model Configuration
- Model: Xenova/parakeet-v2-0.6B
- Task: automatic-speech-recognition
- Device: webgpu
- Caching: Browser IndexedDB via env.useBrowserCache = true

## Build Commands
```bash
npm install           # Install dependencies
npm run build         # Build frontend
npm run dev           # Development server
npm run tauri build   # Full desktop build (requires Rust + system deps)
```
