---
name: Hybrid CG at inflate time is DEAD — contest rules require scorer weights in archive
description: "External libraries and tools can be used... unless they use large artifacts (neural networks), in which case those artifacts should be included in the archive. This applies to the PoseNet and SegNet." Loading scorers at inflate time without including them in archive is rule-violating.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The exact rule from upstream/README.md:
"External libraries and tools can be used and won't count towards
compressed size, unless they use large artifacts (neural networks,
meshes, point clouds, etc.), in which case those artifacts should
be included in the archive and will count towards the compressed size.
This applies to the PoseNet and SegNet."

Including PoseNet (~53MB) + SegNet (~37MB) = 90MB in archive → rate = 60.
This is catastrophic. Hybrid CG at inflate time is NOT contest-compliant.

The renderer-only path is the ONLY viable contest-compliant path.
All inflate-time optimization (CG, TTO, mini-scorer) requires scorer
weights which must be in the archive.
