# cuda lane

This lane is optional.

Suggested uses:
- local teacher-cache generation
- local residual optimization
- optional inflator plugin only if it beats the CPU path cleanly

Do not make CUDA a mandatory dependency for the robust baseline unless the gain is large and stable.
