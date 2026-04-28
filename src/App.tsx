import { useState, useRef, useCallback, useEffect, type FC } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import { getModelConfig, DEFAULT_MODEL } from './models.js';
import { getParakeetModel } from './hub.js';

// Configure transformers.js
env.useBrowserCache = true;
env.allowLocalModels = false;

// Model configuration using models.js
const MODEL_KEY = DEFAULT_MODEL;
const MODEL_CONFIG = getModelConfig(MODEL_KEY);
const MODEL_ID = MODEL_CONFIG?.repoId || 'Unravler/parakeet-tdt-0.6b-v2-onnx';

// Available backends
const AVAILABLE_BACKENDS = ['webgpu', 'cpu'] as const;
type BackendType = typeof AVAILABLE_BACKENDS[number];

// Type for the ASR pipeline
type ASRPipeline = Awaited<ReturnType<typeof pipeline<'automatic-speech-recognition'>>>;

const SUPPORTED_AUDIO = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'];
const SUPPORTED_VIDEO = ['.mp4', '.webm', '.mov', '.avi', '.mkv'];
const SUPPORTED_FORMATS = [...SUPPORTED_AUDIO, ...SUPPORTED_VIDEO];

interface FileInfo {
  name: string;
  size: number;
  file: File;
  duration?: number;
}

interface Stats {
  duration: string;
  processingTime: string;
  wordCount: number;
  wordsPerMinute: number;
  model: string;
}

type ModelStatus = 'idle' | 'downloading' | 'loading' | 'ready' | 'error';

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const secs = ms / 1000;
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const mins = Math.floor(secs / 60);
  const remainingSecs = secs % 60;
  return `${mins}m ${remainingSecs.toFixed(1)}s`;
}

interface ToastProps {
  message: string;
  type: 'success' | 'error';
  onClose: () => void;
}

const Toast: FC<ToastProps> = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`toast ${type}`}>
      {message}
    </div>
  );
};

export default function App() {
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [transcription, setTranscription] = useState('');
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  
  // Check WebGPU availability and preferred backend
  const [preferredBackend, setPreferredBackend] = useState<BackendType>('webgpu');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const transcriberRef = useRef<ASRPipeline | null>(null);
  const startTimeRef = useRef<number>(0);

  // Check WebGPU availability on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        // Try to detect WebGPU support
        // @ts-expect-error navigator.gpu may not exist on all types
        const hasNvidiaGpu = typeof navigator.gpu !== 'undefined';
        
        if (!hasNvidiaGpu) {
          console.warn('[App] WebGPU not available, will use CPU backend');
          setPreferredBackend('cpu');
          setStatusMessage('WebGPU not available - using CPU fallback');
          return;
        }

        // Try to request a GPU adapter to verify actual WebGPU support
        // @ts-expect-error navigator.gpu may not exist
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          console.warn('[App] WebGPU adapter not available, using CPU fallback');
          setPreferredBackend('cpu');
          setStatusMessage('WebGPU adapter unavailable - using CPU fallback');
        } else {
          console.log('[App] WebGPU available, using webgpu backend');
        }
      } catch (error) {
        console.warn('[App] WebGPU check failed:', error);
        setPreferredBackend('cpu');
        setStatusMessage('WebGPU check failed - using CPU fallback');
      }
    };
    checkBackend();
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED_FORMATS.includes(ext)) {
      setToast({ message: 'Unsupported file format', type: 'error' });
      return;
    }

    setSelectedFile({
      name: file.name,
      size: file.size,
      file: file
    });
    setTranscription('');
    setStats(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const loadModel = useCallback(async () => {
    if (transcriberRef.current || modelStatus === 'loading' || modelStatus === 'downloading') return;

    setModelStatus('downloading');
    setProgress(0);
    setStatusMessage('Checking model availability...');

    try {
      // First, verify model files are available using hub.js
      // This queries the HuggingFace API to list available files
      setModelStatus('loading');
      setStatusMessage(`Fetching model info from HuggingFace...`);
      
      const progressCallback = (p: { status?: string; progress?: number; file?: string }) => {
        if (p.status) {
          setStatusMessage(p.status);
        }
        if (p.file) {
          setStatusMessage(`Downloading ${p.file}...`);
        }
        if (p.progress !== undefined) {
          setProgress(Math.round(p.progress * 100));
        }
      };

      // Use getParakeetModel to verify model is available
      // and get the file URLs with proper quantization
      const modelInfo = await getParakeetModel(MODEL_KEY, {
        backend: preferredBackend,
        progress: progressCallback,
      });
      
      console.log('[Model] Model info:', {
        model: MODEL_KEY,
        repo: MODEL_ID,
        files: modelInfo.filenames,
        quant: modelInfo.quantisation,
        preprocessor: modelInfo.preprocessorBackend,
      });

      setStatusMessage(`Loading model with ${preferredBackend} backend...`);

      // Load the ASR pipeline with the verified model
      transcriberRef.current = await pipeline(
        'automatic-speech-recognition',
        MODEL_ID,
        {
          progress_callback: progressCallback,
          device: preferredBackend,
        }
      );

      // Load the ASR pipeline with the verified model
      // Add retries for network errors
      let lastError: Error | null = null;
      const maxRetries = 3;
      
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          setStatusMessage(`Loading model (attempt ${attempt}/${maxRetries})...`);
          
          transcriberRef.current = await pipeline(
            'automatic-speech-recognition',
            MODEL_ID,
            {
              progress_callback: progressCallback,
              device: preferredBackend,
            }
          );
          
          setModelStatus('ready');
          setProgress(100);
          setStatusMessage('Model ready');
          setToast({ message: 'Model loaded successfully', type: 'success' });
          return; // Success, exit the function
        } catch (err) {
          lastError = err instanceof Error ? err : new Error(String(err));
          console.error(`[Model] Attempt ${attempt} failed:`, lastError.message);
          
          if (attempt < maxRetries) {
            // Check if it's a network error that might succeed on retry
            if (lastError.message.includes('fetch') || lastError.message.includes('JSON') || lastError.message.includes('network') || lastError.message.includes('Failed to fetch')) {
              setStatusMessage(`Retrying in ${attempt * 2}s...`);
              await new Promise(r => setTimeout(r, attempt * 2000));
              continue;
            }
            // For non-network errors, don't retry
            break;
          }
        }
      }
      
      throw lastError;
    } catch (error) {
      console.error('[Model] Loading error:', error);
      
      // If WebGPU failed, try with CPU fallback
      if (preferredBackend === 'webgpu' && error instanceof Error) {
        console.warn('[Model] WebGPU failed, retrying with CPU...');
        setPreferredBackend('cpu');
        setModelStatus('idle');
        setProgress(0);
        setStatusMessage('WebGPU failed, retrying with CPU...');
        
        try {
          transcriberRef.current = await pipeline(
            'automatic-speech-recognition',
            MODEL_ID,
            {
              progress_callback: (p: { status?: string; progress?: number; file?: string }) => {
                if (p.status) setStatusMessage(p.status);
                if (p.file) setStatusMessage(`Downloading ${p.file}...`);
                if (p.progress !== undefined) setProgress(Math.round(p.progress * 100));
              },
              device: 'cpu',
            }
          );
          setModelStatus('ready');
          setProgress(100);
          setStatusMessage('Model ready (CPU mode)');
          setToast({ message: 'Model loaded with CPU fallback', type: 'success' });
        } catch (cpuError) {
          console.error('[Model] CPU fallback error:', cpuError);
          setModelStatus('error');
          setStatusMessage(cpuError instanceof Error ? cpuError.message : 'Failed to load model');
          setToast({ message: 'Failed to load model', type: 'error' });
        }
        return;
      }
      
      setModelStatus('error');
      setStatusMessage(error instanceof Error ? error.message : 'Failed to load model');
      setToast({ message: 'Failed to load model', type: 'error' });
    }
  }, [preferredBackend]);

  const transcribe = useCallback(async () => {
    if (!selectedFile || !transcriberRef.current) return;

    setIsTranscribing(true);
    setTranscription('');
    setStats(null);
    startTimeRef.current = Date.now();

    try {
      setStatusMessage('Reading audio file...');
      
      setStatusMessage('Processing audio...');
      
      // Transcribe using transformers.js pipeline - use file URL
      const audioUrl = URL.createObjectURL(selectedFile.file);
      const result = await transcriberRef.current(audioUrl, {
        max_new_tokens: 48000,
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true,
      });
      
      // Clean up the URL
      URL.revokeObjectURL(audioUrl);
      
      const processingTime = Date.now() - startTimeRef.current;
      const output = result as { text: string };
      const text = output.text;
      
      setTranscription(text);
      
      // Estimate duration from word count (rough estimate)
      const wordCount = text.split(/\s+/).filter((w: string) => w.length > 0).length;
      const estimatedDuration = wordCount * 0.15; // rough estimate: ~0.15s per word

      setStats({
        duration: formatDuration(estimatedDuration),
        processingTime: formatTime(processingTime),
        wordCount,
        wordsPerMinute: Math.round(wordCount / (processingTime / 60000)),
        model: MODEL_ID
      });
      
      setStatusMessage('Transcription complete');
      setToast({ message: 'Transcription complete', type: 'success' });
    } catch (error) {
      console.error('Transcription error:', error);
      setStatusMessage(error instanceof Error ? error.message : 'Transcription failed');
      setToast({ message: 'Transcription failed', type: 'error' });
    } finally {
      setIsTranscribing(false);
    }
  }, [selectedFile]);

  const copyToClipboard = useCallback(() => {
    if (!transcription) return;
    navigator.clipboard.writeText(transcription);
    setToast({ message: 'Copied to clipboard', type: 'success' });
  }, [transcription]);

  const downloadText = useCallback(() => {
    if (!transcription) return;
    
    const blob = new Blob([transcription], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedFile 
      ? selectedFile.name.replace(/\.[^.]+$/, '.txt')
      : 'transcription.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setToast({ message: 'Downloaded', type: 'success' });
  }, [transcription, selectedFile]);

  const getStatusDotClass = () => {
    if (modelStatus === 'ready') return 'status-dot ready';
    if (modelStatus === 'loading' || modelStatus === 'downloading') return 'status-dot loading';
    if (modelStatus === 'error') return 'status-dot error';
    return 'status-dot';
  };

  const getStatusText = () => {
    switch (modelStatus) {
      case 'idle': return `Not loaded (${preferredBackend})`;
      case 'downloading': return 'Preparing...';
      case 'loading': return 'Loading...';
      case 'ready': return 'Ready';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Tauri Transcribe</h1>
        <div className="status-badge">
          <span className={getStatusDotClass()} />
          <span>{getStatusText()}</span>
        </div>
      </header>

      <section className="card">
        <h2 className="card-title">File Selection</h2>
        <div
          className={`file-drop-zone ${isDragOver ? 'drag-over' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <p>Drop audio/video file here or click to browse</p>
          <span className="formats">
            Supported: {SUPPORTED_AUDIO.join(', ')} | {SUPPORTED_VIDEO.join(', ')}
          </span>
          <input
            ref={fileInputRef}
            type="file"
            className="file-input"
            accept={SUPPORTED_FORMATS.join(',')}
            onChange={handleFileInputChange}
          />
        </div>
        
        {selectedFile && (
          <div className="selected-file">
            <div className="selected-file-info">
              <span className="selected-file-name">{selectedFile.name}</span>
              <span className="selected-file-size">{formatFileSize(selectedFile.size)}</span>
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Model</h2>
        <div className="model-controls">
          <button
            className="btn btn-primary"
            onClick={loadModel}
            disabled={modelStatus === 'loading' || modelStatus === 'downloading'}
          >
            {(modelStatus === 'loading' || modelStatus === 'downloading') ? (
              <>
                <span className="spinner" />
                Loading...
              </>
            ) : (
              'Load Model'
            )}
          </button>
          
          <button
            className="btn btn-success"
            onClick={transcribe}
            disabled={!selectedFile || modelStatus !== 'ready' || isTranscribing}
          >
            {isTranscribing ? (
              <>
                <span className="spinner" />
                Transcribing...
              </>
            ) : (
              'Transcribe'
            )}
          </button>

          {(modelStatus === 'loading' || modelStatus === 'downloading') && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <p className="progress-text">{statusMessage}</p>
            </div>
          )}
        </div>
      </section>

      <section className="card transcription-area">
        <h2 className="card-title">Transcription</h2>
        <textarea
          className="text-output"
          placeholder="Transcription will appear here..."
          value={transcription}
          onChange={(e) => setTranscription(e.target.value)}
          readOnly={isTranscribing}
        />
        
        <div className="transcription-controls">
          <button
            className="btn btn-secondary"
            onClick={copyToClipboard}
            disabled={!transcription}
          >
            Copy to Clipboard
          </button>
          <button
            className="btn btn-secondary"
            onClick={downloadText}
            disabled={!transcription}
          >
            Download .txt
          </button>
        </div>
      </section>

      {stats && (
        <section className="card">
          <h2 className="card-title">Statistics</h2>
          <div className="stats-panel">
            <div className="stat-item">
              <p className="stat-label">Duration</p>
              <p className="stat-value">{stats.duration}</p>
            </div>
            <div className="stat-item">
              <p className="stat-label">Processing Time</p>
              <p className="stat-value">{stats.processingTime}</p>
            </div>
            <div className="stat-item">
              <p className="stat-label">Words</p>
              <p className="stat-value">{stats.wordCount}</p>
            </div>
            <div className="stat-item">
              <p className="stat-label">Words/Min</p>
              <p className="stat-value">{stats.wordsPerMinute}</p>
            </div>
            <div className="stat-item">
              <p className="stat-label">Model</p>
              <p className="stat-value" style={{ fontSize: '12px' }}>{stats.model}</p>
            </div>
          </div>
        </section>
      )}

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}

      <footer className="footer">
        Using Unravler/parakeet-tdt-0.6b-v2-onnx with WebGPU acceleration
      </footer>
    </div>
  );
}
