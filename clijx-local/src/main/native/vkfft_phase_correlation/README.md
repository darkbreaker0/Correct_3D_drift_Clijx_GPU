# VkFFT phase correlation (native build)

This native library provides GPU phase correlation using VkFFT (OpenCL backend).
It is required by `net.haesleinhuepf.clijx.fft.VkFFTPhaseCorrelation`.

## Build (Windows, Visual Studio + CMake)

1. Install a C++ compiler and CMake.
2. Ensure OpenCL headers and runtime are installed (GPU drivers).
3. Configure and build:

```bash
cmake -S . -B build
cmake --build build --config Release
```

The output library is `clijx_vkfft.dll` (name may include configuration suffix).

## Runtime loading

Set `CLIJX_VKFFT_PATH` to the full path of the compiled library, or place it on
your `PATH`. The Java code calls `System.loadLibrary("clijx_vkfft")` by default.
