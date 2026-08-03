export interface BrowserModelManifest {
  schemaVersion: 1;
  modelUrl: string;
  sha256: string;
  featureSpec: "alphasho-119-plane-v1";
  policySize: number;
  modelVersion: string;
}

// Reserved for a future ONNX/WebGPU player. The first static release deliberately
// ships no model and never exposes an unavailable neural-network menu item.
