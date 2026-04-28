/**
 * HuggingFace Hub utilities for loading Parakeet ONNX models.
 * Downloads models and caches them in browser IndexedDB.
 */

import { getModelConfig } from './models.js';

/** @typedef {import('./models.js').ModelConfig} ModelConfig */

const DB_NAME = 'parakeet-cache-db';
const STORE_NAME = 'file-store';

let dbPromise = null;

// Cache for repo file listings
const repoFileCache = new Map();

const QUANT_SUFFIX = {
  int8: '.int8.onnx',
  fp16: '.fp16.onnx',
  fp32: '.onnx',
};

/**
 * @param {string} baseName
 * @param {('int8'|'fp32'|'fp16')} quant
 * @returns {string}
 */
function getQuantizedModelName(baseName, quant) {
  const suffix = QUANT_SUFFIX[quant];
  if (!suffix) {
    throw new Error(`[Hub] Unknown quantization '${quant}'`);
  }
  return `${baseName}${suffix}`;
}

/**
 * Encode HF repo path
 * @param {string} repoId
 * @returns {string}
 */
function encodeRepoPath(repoId) {
  return String(repoId || '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

/**
 * Normalize path entry
 * @param {string} path
 * @returns {string}
 */
function normalizeRepoPath(path) {
  if (typeof path !== 'string') return '';
  return path.replace(/^\.\//, '').replace(/\\/g, '/');
}

/**
 * Parse file listing from HF API
 * @param {unknown} payload
 * @returns {string[]}
 */
function parseRepoListingPayload(payload) {
  if (Array.isArray(payload)) {
    return payload
      .filter((entry) => entry?.type === 'file' && typeof entry?.path === 'string')
      .map((entry) => normalizeRepoPath(entry.path));
  }
  if (payload && typeof payload === 'object' && Array.isArray(payload.siblings)) {
    return payload.siblings
      .map((entry) => normalizeRepoPath(entry?.rfilename))
      .filter(Boolean);
  }
  return [];
}

/**
 * Check if repo contains a file
 * @param {string[]|null} repoFiles
 * @param {string} filename
 * @returns {boolean}
 */
function repoHasFile(repoFiles, filename) {
  if (!repoFiles) return false;
  const target = normalizeRepoPath(filename);
  return repoFiles.some((path) => path === target || path.endsWith(`/${target}`));
}

/**
 * List model repository files
 * @param {string} repoId
 * @param {string} [revision='main']
 * @returns {Promise<string[]|null>}
 */
async function listRepoFiles(repoId, revision = 'main') {
  const cacheKey = `${repoId}@${revision}`;
  if (repoFileCache.has(cacheKey)) return repoFileCache.get(cacheKey);

  const encodedRepoId = encodeRepoPath(repoId);
  const encodedRevision = encodeURIComponent(revision);
  const treeUrl = `https://huggingface.co/api/models/${encodedRepoId}/tree/${encodedRevision}?recursive=1`;
  const modelUrl = `https://huggingface.co/api/models/${encodedRepoId}?revision=${encodedRevision}`;

  try {
    const resp = await fetch(treeUrl);
    if (!resp.ok) throw new Error(`Tree API: ${resp.status}`);
    const files = parseRepoListingPayload(await resp.json());
    repoFileCache.set(cacheKey, files);
    return files;
  } catch (treeErr) {
    console.warn('[Hub] Tree API failed, trying metadata', treeErr);
  }

  try {
    const resp = await fetch(modelUrl);
    if (!resp.ok) throw new Error(`Metadata API: ${resp.status}`);
    const files = parseRepoListingPayload(await resp.json());
    repoFileCache.set(cacheKey, files);
    return files;
  } catch (metadataErr) {
    console.warn('[Hub] Metadata API failed', metadataErr);
    return null;
  }
}

/**
 * Get (or initialize) IndexedDB
 * @returns {Promise<IDBDatabase>}
 */
function getDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onerror = () => reject('Error opening IndexedDB');
      request.onsuccess = () => resolve(request.result);
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };
    });
  }
  return dbPromise;
}

/**
 * Get cached file from IndexedDB
 * @param {string} repoId
 * @param {string} filename
 * @returns {Promise<Blob|null>}
 */
async function getCachedFile(repoId, filename) {
  try {
    const db = await getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const key = `${repoId}/${filename}`;
      const request = store.get(key);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result || null);
    });
  } catch {
    return null;
  }
}

/**
 * Cache file to IndexedDB
 * @param {string} repoId
 * @param {string} filename
 * @param {Blob} blob
 */
async function cacheFile(repoId, filename, blob) {
  try {
    const db = await getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const key = `${repoId}/${filename}`;
      const request = store.put(blob, key);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  } catch (err) {
    console.warn('[Hub] Failed to cache file:', err);
  }
}

/**
 * Get model file with caching
 * @param {string} repoId
 * @param {string} filename
 * @param {Object} [options]
 * @param {('int8'|'fp32'|'fp16')} [options.quant]
 * @param {string} [options.revision]
 * @param {(progress: {loaded: number, total: number, file: string}) => void} [options.progress]
 * @returns {Promise<string>} Blob URL
 */
async function getModelFile(repoId, filename, options = {}) {
  const { progress } = options;

  // Check cache first
  const cached = await getCachedFile(repoId, filename);
  if (cached) {
    console.log(`[Hub] Using cached: ${filename}`);
    return URL.createObjectURL(cached);
  }

  // Download from HuggingFace
  const encodedRepoId = encodeRepoPath(repoId);
  const revision = options.revision || 'main';
  const url = `https://huggingface.co/${encodedRepoId}/resolve/${revision}/${filename}`;

  console.log(`[Hub] Downloading: ${filename}`);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${filename}: ${response.status}`);
  }

  const total = Number(response.headers.get('content-length') || 0);
  let loaded = 0;

  // Handle progress for streaming response
  if (progress && total) {
    const reader = response.body?.getReader();
    if (reader) {
      const chunks = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        progress({ loaded, total, file: filename });
      }
      const blob = new Blob(chunks);
      await cacheFile(repoId, filename, blob);
      return URL.createObjectURL(blob);
    }
  }

  // Fallback: simple download
  const blob = await response.blob();
  await cacheFile(repoId, filename, blob);
  return URL.createObjectURL(blob);
}

/**
 * Create model component config
 * @param {'encoder'|'decoder'} name
 * @param {'int8'|'fp32'|'fp16'} quant
 * @returns {Object}
 */
function createComponent(name, quant) {
  const baseName = name === 'encoder' ? 'encoder-model' : 'decoder_joint-model';
  const key = name === 'encoder' ? 'encoderUrl' : 'decoderUrl';
  return {
    name,
    key,
    baseName,
    quant,
    filename: getQuantizedModelName(baseName, quant),
  };
}

/**
 * Validate FP16 component exists
 * @param {Object} component
 * @param {string} repoId
 * @param {string[]|null} repoFiles
 */
function validateRequestedFp16Component(component, repoId, repoFiles) {
  if (component.quant !== 'fp16' || repoFiles === null) return;

  const fp16Name = getQuantizedModelName(component.baseName, 'fp16');
  const fp32Name = getQuantizedModelName(component.baseName, 'fp32');

  if (repoHasFile(repoFiles, fp16Name)) return;

  if (repoHasFile(repoFiles, fp32Name)) {
    throw new Error(
      `[Hub] ${component.name} FP16 missing in ${repoId} (found ${fp32Name}). Use fp32 explicitly.`
    );
  }

  throw new Error(`[Hub] Missing ${component.name} model: ${fp16Name}`);
}

/**
 * Get all required download files
 * @param {Object} components
 * @param {'js'|'onnx'} preprocessorBackend
 * @param {string} preprocessor
 * @returns {Array}
 */
function buildRequiredDownloads(components, preprocessorBackend, preprocessor) {
  const files = [
    { key: components.encoder.key, name: components.encoder.filename, componentName: 'encoder', optional: false },
    { key: components.decoder.key, name: components.decoder.filename, componentName: 'decoder', optional: false },
    { key: 'tokenizerUrl', name: 'vocab.txt', optional: false },
  ];

  if (preprocessorBackend !== 'js') {
    files.push({ key: 'preprocessorUrl', name: `${preprocessor}.onnx`, optional: false });
  }

  return files;
}

/**
 * Get optional external data files
 * @param {Object} components
 * @param {string[]|null} repoFiles
 * @returns {Array}
 */
function buildOptionalExternalDataDownloads(components, repoFiles) {
  const candidates = [
    { key: 'encoderDataUrl', name: `${components.encoder.filename}.data`, optional: true },
    { key: 'decoderDataUrl', name: `${components.decoder.filename}.data`, optional: true },
  ];
  if (repoFiles === null) return [];
  return candidates.filter((entry) => repoHasFile(repoFiles, entry.name));
}

/**
 * Get Parakeet model files.
 * @param {string} repoIdOrModelKey - HF repo ID or model key
 * @param {Object} [options]
 * @param {('int8'|'fp32'|'fp16')} [options.encoderQuant]
 * @param {('int8'|'fp32'|'fp16')} [options.decoderQuant]
 * @param {('js'|'onnx')} [options.preprocessorBackend]
 * @param {(progress: {loaded: number, total: number, file: string}) => void} [options.progress]
 * @returns {Promise<Object>}
 */
export async function getParakeetModel(repoIdOrModelKey, options = {}) {
  const modelConfig = getModelConfig(repoIdOrModelKey);
  const repoId = modelConfig?.repoId || repoIdOrModelKey;
  const defaultPreprocessor = modelConfig?.preprocessor || 'nemo128';

  const {
    encoderQuant = 'int8',
    decoderQuant = 'int8',
    preprocessorBackend = 'js',
    progress,
  } = options;

  // Force fp32 for encoder on WebGPU (int8 not supported)
  let backend = options.backend || 'webgpu';
  let encoderQ = encoderQuant;
  // Handle cpu backend - treat as non-webgpu
  if (backend === 'cpu') {
    backend = 'wasm';
  }
  if (backend.startsWith('webgpu') && encoderQ === 'int8') {
    console.warn('[Hub] Forcing encoder to fp32 on WebGPU (int8 unsupported)');
    encoderQ = 'fp32';
  }

  const components = {
    encoder: createComponent('encoder', encoderQ),
    decoder: createComponent('decoder', decoderQuant),
  };

  const repoFiles = await listRepoFiles(repoId, options.revision || 'main');

  validateRequestedFp16Component(components.encoder, repoId, repoFiles);
  validateRequestedFp16Component(components.decoder, repoId, repoFiles);

  const requiredFiles = buildRequiredDownloads(components, preprocessorBackend, defaultPreprocessor);

  const results = {
    urls: {},
    filenames: {
      encoder: components.encoder.filename,
      decoder: components.decoder.filename,
    },
    quantisation: {
      encoder: components.encoder.quant,
      decoder: components.decoder.quant,
    },
    modelConfig: modelConfig || null,
    preprocessorBackend,
  };

  for (const file of requiredFiles) {
    try {
      results.urls[file.key] = await getModelFile(repoId, file.name, { ...options, progress });
    } catch (err) {
      if (!file.componentName) throw err;

      const component = components[file.componentName];
      if (component.quant === 'fp16') {
        throw new Error(
          `[Hub] ${component.name} FP16 failed (${file.name}). Use fp32 explicitly.`
        );
      }
      throw err;
    }
  }

  const optionalFiles = buildOptionalExternalDataDownloads(components, repoFiles);
  for (const file of optionalFiles) {
    try {
      results.urls[file.key] = await getModelFile(repoId, file.name, { ...options, progress });
    } catch {
      console.warn(`[Hub] Optional file not found: ${file.name}`);
      results.urls[file.key] = null;
    }
  }

  return results;
}