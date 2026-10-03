import os

# XLA's CPU optimizer takes minutes on the vmapped step; level 0 compiles in about a second. Tests only need
# correctness, and GPU compilation is unaffected. Must be set before jax is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_backend_optimization_level=0")
