"""
titan csrc package.

Importing this package does NOT automatically load the compiled CUDA
extension — you must load it explicitly with:

    import titan_kernels  # loads TORCH_LIBRARY registrations

This package provides Python wrappers and utilities that sit on top of the
compiled ops.  Subpackages:

    csrc.neural_memory  — NeuralMemory patch helpers
    csrc.block_ell      — (future) Python autograd Function for swiglu ops
    csrc.loop           — (future) Python wrapper for persistent_mlp_step
"""
