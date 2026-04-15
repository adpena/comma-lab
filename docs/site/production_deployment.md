# Production Deployment

## From contest to fleet

The comma.ai video compression challenge models a real engineering problem: compressing dashcam video while preserving what the driving stack actually needs. Our system --- a 287K-parameter renderer that generates frames from semantic masks, trained end-to-end against comma's SegNet and PoseNet scorers --- achieves a score of 0.43 (vs. 0.60 for the next compliant submission). The architecture maps directly to comma's production data pipeline.

## Where it fits

comma's fleet of 250,000+ devices captures petabytes of driving video, uploaded over cellular and stored on servers for training data. Our renderer compresses 60 seconds of video into a 150 KB archive (masks + weights) --- a 17x reduction over H.265, 200x over raw video. Reconstruction takes 3 seconds on a T4 GPU with 0.22% SegNet cross-entropy disagreement and 3.1% PoseNet L2 error. The 150 KB archive includes everything needed to reconstruct: no external model weights, no scorer loading, no dependencies beyond a standard PyTorch runtime.

## Scalability

The system has a compute-quality knob. Renderer-only reconstruction (3 seconds, score 0.87) is cheap enough for real-time mobile playback on Snapdragon hardware. Adding test-time optimization (40 minutes on a 4090, score 0.43) pre-computes pixel corrections that are applied at decompress time as a single additive step. The compression is expensive once; every subsequent read is fast. For fleet archival, the amortized cost is approximately $0.001 per route for renderer-only, or $0.17 per route for full TTO on a 4090.

## Details

For a full analysis --- including comma hardware specs, per-operation latency estimates, cost modeling for fleet-scale deployment, and a discussion of limitations (single-video evaluation, real-time encoding, fleet diversity) --- see [Section 5 of the paper](../paper/05_production.md).
