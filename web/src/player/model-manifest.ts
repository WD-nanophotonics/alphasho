import type { NnueManifest } from './nnue-config';

export interface LegacyCnnModelManifest {
  schemaVersion: 1;
  modelUrl: string;
  sha256: string;
  featureSpec: "alphasho-119-plane-v1";
  policySize: number;
  modelVersion: string;
}

// Distinct formats: a HalfKP evaluation network must never be interpreted as
// the legacy 119-plane policy/value network reserved for ONNX/WebGPU.
export type BrowserModelManifest = NnueManifest | LegacyCnnModelManifest;
