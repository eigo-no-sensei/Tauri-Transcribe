/**
 * Hub module type declarations
 */

import type { ModelConfig } from './models.d';

export interface ProgressCallback {
  loaded: number;
  total: number;
  file: string;
}

export interface ModelQuantisation {
  encoder: 'int8' | 'fp32' | 'fp16';
  decoder: 'int8' | 'fp32' | 'fp16';
}

export interface GetParakeetModelOptions {
  encoderQuant?: 'int8' | 'fp32' | 'fp16';
  decoderQuant?: 'int8' | 'fp32' | 'fp16';
  preprocessorBackend?: 'js' | 'onnx';
  backend?: 'webgpu' | 'webgpu-hybrid' | 'webgpu-strict' | 'wasm' | 'cpu';
  revision?: string;
  progress?: (progress: ProgressCallback) => void;
}

export interface ParakeetModelResult {
  urls: {
    encoderUrl?: string;
    decoderUrl?: string;
    tokenizerUrl?: string;
    preprocessorUrl?: string;
    encoderDataUrl?: string | null;
    decoderDataUrl?: string | null;
  };
  filenames: {
    encoder: string;
    decoder: string;
  };
  quantisation: ModelQuantisation;
  modelConfig: ModelConfig | null;
  preprocessorBackend: 'js' | 'onnx';
}

export declare function getParakeetModel(
  repoIdOrModelKey: string,
  options?: GetParakeetModelOptions
): Promise<ParakeetModelResult>;