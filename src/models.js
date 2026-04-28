/**
 * Model configurations for supported Parakeet variants.
 * Centralizes model metadata for easier maintenance.
 */

// Language display names
export const LANGUAGE_NAMES = {
  en: 'English',
  fr: 'French',
  de: 'German',
  es: 'Spanish',
  it: 'Italian',
  pt: 'Portuguese',
  nl: 'Dutch',
  pl: 'Polish',
  ru: 'Russian',
  uk: 'Ukrainian',
  ja: 'Japanese',
  ko: 'Korean',
  zh: 'Chinese',
};

/**
 * @typedef {Object} ModelConfig
 * @property {string} repoId - HuggingFace repository ID
 * @property {string} displayName - Human-readable name
 * @property {string[]} languages - Supported languages
 * @property {string} defaultLanguage - Default language
 * @property {number} vocabSize - Vocabulary size
 * @property {number} featuresSize - Mel spectrogram features (80 or 128)
 * @property {string} preprocessor - Default preprocessor variant
 * @property {number} subsampling - Subsampling factor
 * @property {number} predHidden - Prediction network hidden size
 * @property {number} predLayers - Prediction network layers
 */

/**
 * Supported model configurations
 * @type {Object.<string, ModelConfig>}
 */
export const MODELS = {
  'parakeet-tdt-0.6b-v2': {
    repoId: 'istupakov/parakeet-tdt-0.6b-v2-onnx',
    displayName: 'Parakeet TDT 0.6B v2 (English)',
    languages: ['en'],
    defaultLanguage: 'en',
    vocabSize: 1025,
    featuresSize: 128,
    preprocessor: 'nemo128',
    subsampling: 8,
    predHidden: 640,
    predLayers: 2,
  },
};

/** Default model */
export const DEFAULT_MODEL = 'parakeet-tdt-0.6b-v2';

/**
 * Get model configuration by key or repo ID
 * @param {string} modelKeyOrRepoId
 * @returns {ModelConfig|null}
 */
export function getModelConfig(modelKeyOrRepoId) {
  if (MODELS[modelKeyOrRepoId]) {
    return MODELS[modelKeyOrRepoId];
  }
  for (const [key, config] of Object.entries(MODELS)) {
    if (config.repoId === modelKeyOrRepoId) {
      return config;
    }
  }
  return null;
}

/**
 * Get model key from repo ID
 * @param {string} repoId
 * @returns {string|null}
 */
export function getModelKeyFromRepoId(repoId) {
  for (const [key, config] of Object.entries(MODELS)) {
    if (config.repoId === repoId) {
      return key;
    }
  }
  return null;
}

/**
 * Check if model supports a language
 * @param {string} modelKeyOrRepoId
 * @param {string} language
 * @returns {boolean}
 */
export function supportsLanguage(modelKeyOrRepoId, language) {
  const config = getModelConfig(modelKeyOrRepoId);
  if (!config) return false;
  return config.languages.includes(language.toLowerCase());
}

/**
 * List all available model keys
 * @returns {string[]}
 */
export function listModels() {
  return Object.keys(MODELS);
}

/**
 * Get language display name
 * @param {string} langCode
 * @returns {string}
 */
export function getLanguageName(langCode) {
  return LANGUAGE_NAMES[langCode.toLowerCase()] || langCode;
}