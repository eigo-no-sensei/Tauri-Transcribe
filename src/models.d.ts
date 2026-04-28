/**
 * Model configurations for Parakeet variants
 */

export declare const LANGUAGE_NAMES: Record<string, string>;

export interface ModelConfig {
  repoId: string;
  displayName: string;
  languages: string[];
  defaultLanguage: string;
  vocabSize: number;
  featuresSize: number;
  preprocessor: string;
  subsampling: number;
  predHidden: number;
  predLayers: number;
}

export declare const MODELS: Record<string, ModelConfig>;
export declare const DEFAULT_MODEL: string;

export declare function getModelConfig(modelKeyOrRepoId: string): ModelConfig | null;
export declare function getModelKeyFromRepoId(repoId: string): string | null;
export declare function supportsLanguage(modelKeyOrRepoId: string, language: string): boolean;
export declare function listModels(): string[];
export declare function getLanguageName(langCode: string): string;