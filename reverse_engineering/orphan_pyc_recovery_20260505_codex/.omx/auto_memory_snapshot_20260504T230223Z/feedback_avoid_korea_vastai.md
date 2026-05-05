---
name: Avoid Korea Vast.ai for 4090s
description: Korea region 4090 instances ship a base image where torch can't see CUDA on the 4090 hardware (driver 12.4 host, "forward compatibility on non-supported HW" error 804).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule:** When searching `vastai search offers` for RTX_4090, EXPLICITLY EXCLUDE Korea/KR via `geolocation!=KR` filter or pick a US/EU instance manually.

**Why:** 2026-04-26: spun up instance 35602310 (id=26668774) in South Korea. Base image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` reported `torch.cuda.is_available() == False` with `Error 804: forward compatibility was attempted on non supported HW`. Same image works fine on US/Texas instance 35602738 (id=29904360). Wasted ~30 min of $0.28/hr trying to recover. Destroyed and switched to Texas, where it just works.

**How to apply:**
- Default Vast.ai 4090 search filter: `gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30 num_gpus=1 verified=True geolocation!=KR`
- US (Texas, California, Virginia) and EU (Netherlands, Bulgaria) confirmed working as of 2026-04-26.
- If a Korea instance ever appears in search results, skip it — even if it's the cheapest. The 30 minutes of wasted time + reup costs more than the price differential.
