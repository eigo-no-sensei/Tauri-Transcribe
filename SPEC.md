# Tauri-Transcribe Specification

## Project Overview
- **Project Name**: Tauri-Transcribe
- **Type**: Desktop Transcription Application
- **Core Functionality**: Audio/Video to text transcription using transformers.js with WebGPU acceleration
- **Target Users**: Content creators, journalists, researchers needing local transcription

## UI/UX Specification

### Layout Structure
- **Single Window Application** with responsive layout
- **Header**: App title and model status indicator
- **Main Content Area**:
  - File browser section (top)
  - Model controls section (middle)
  - Transcription output section (bottom)
- **Footer**: Stats display area

### Responsive Breakpoints
- Desktop-first design (min 800x600)
- Graceful scaling for smaller windows

### Visual Design

#### Color Palette
- **Background**: `#0f0f0f` (near black)
- **Surface**: `#1a1a1a` (dark gray)
- **Surface Elevated**: `#252525` (lighter gray)
- **Primary**: `#3b82f6` (blue)
- **Primary Hover**: `#2563eb`
- **Accent**: `#10b981` (emerald green)
- **Text Primary**: `#f5f5f5` (off-white)
- **Text Secondary**: `#a3a3a3` (gray)
- **Border**: `#333333`
- **Error**: `#ef4444` (red)
- **Success**: `#22c55e` (green)

#### Typography
- **Font Family**: `"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
- **Title**: 20px, font-weight 600
- **Headings**: 16px, font-weight 600
- **Body**: 14px, font-weight 400
- **Small/Labels**: 12px, font-weight 500

#### Spacing System
- Base unit: 4px
- Small: 8px
- Medium: 16px
- Large: 24px
- XLarge: 32px

#### Visual Effects
- Rounded corners: 8px (cards), 6px (buttons), 4px (inputs)
- Box shadows: `0 2px 8px rgba(0,0,0,0.3)` (elevated elements)
- Transitions: 150ms ease-out

### Components

#### File Browser
- Drag-and-drop zone with dashed border
- File input button (styled)
- Supported formats indicator: MP3, WAV, OGG, MP4, WEBM, MOV
- Selected file display with name and size

#### Model Controls
- Load Model button (primary action)
- Model status indicator (Not Loaded / Downloading / Loading / Ready)
- Progress bar for model download/load

#### Transcription Area
- Large text area for displaying transcription
- Copy to clipboard button (with success feedback)
- Download as .txt button

#### Stats Panel
- Transcription duration
- Processing time
- Words per minute
- Model info display

## Functionality Specification

### Core Features

1. **File Loading**
   - Accept audio files: .mp3, .wav, .ogg, .flac, .m4a
   - Accept video files: .mp4, .webm, .mov, .avi, .mkv
   - Display file name, size, duration after loading
   - File validation with clear error messages

2. **Model Management**
   - Load Unravler/parakeet-tdt-0.6b-v2-onnx model from transformers.js
   - WebGPU backend
   - Model caching in app data directory
   - Progress indication during download/load

3. **Transcription**
   - Real-time progress updates
   - Cancel capability
   - Handle long audio files
   - Output to editable text area

4. **Export Features**
   - Copy to clipboard with notification
   - Download as .txt file
   - Original filename with .txt extension

5. **Statistics**
   - Media duration
   - Processing time
   - Words transcribed
   - Words per minute calculation

### User Interactions
- Drag and drop or click to select file
- Single click to load model (if not loaded)
- Click to start transcription
- Click copy/download after transcription

### Edge Cases
- No WebGPU available: Show error message
- Model download fails: Retry option
- Invalid file format: Clear error message
- Empty audio: Show warning
- Transcription fails: Error display with retry

## Technical Specification

### Stack
- **Framework**: Tauri 2.x
- **Frontend**: React 18 with TypeScript
- **Styling**: Plain CSS with CSS variables
- **ML**: transformers.js v4 (@huggingface/transformers)
- **Build**: Vite

### Model Configuration
- **Model**: Unravler/parakeet-tdt-0.6b-v2-onnx
- **Backend**: WebGPU
- **Task**: automatic-speech-recognition

### Caching Strategy
- Default: Browser cache via IndexedDB (transformers.js automatic caching)
- Option: Bundle with build (production)

## Build Instructions

### Prerequisites
- Node.js 18+
- Rust 1.70+
- For Linux desktop builds: pkg-config, libglib2.0-dev, libgtk-3-dev, libsoup-3.0-dev, libwebkit2gtk-4.1-dev

### Build Commands
```bash
# Install dependencies
npm install

# Build frontend
npm run build

# Build Tauri (requires system dependencies for Linux)
npm run tauri build
```

## Acceptance Criteria

1. App launches without errors
2. Can load audio/video file via drag-drop or file picker
3. Model loads successfully with progress indication
4. Transcription runs and displays text
5. Copy to clipboard works
6. Download as .txt works
7. Stats display after completion
8. Clean, dark theme interface
9. Responsive to window resizing
10. Works without internet after model cached