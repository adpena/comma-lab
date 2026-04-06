# mojo lane

Mojo is kept as an experimental lane in this starter pack.

Use it only for narrow benchmarked kernels first:
- upsample
- ROI patch apply
- tiny numeric transforms

Do not make Mojo a required dependency for the mainline lab until it wins a measured hot-path benchmark and the deployment story is clean.

Suggested first step:

```bash
pixi add mojo
pixi shell
mojo hello.mojo
```
