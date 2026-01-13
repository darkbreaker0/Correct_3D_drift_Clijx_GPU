# Changes vs Correct_3D_drift.py

This file summarizes the differences in `Correct_3D_drift_clij2.py` compared to the original Fiji script `Correct_3D_drift.py`.

## Core changes
- **GPU translation path:** frame translation is performed with CLIJ2 (`translate2D/3D` or `affineTransform2D/3D`) instead of pure CPU-only translation.
- **GPU phase correlation (when available):** drift estimation first tries `PhaseCorrelationFFT` via CLIJX/VkFFT on the GPU, then falls back to CPU phase correlation if the native library is unavailable.
- **Native library handling:** the script checks `CLIJX_VKFFT_PATH` (or a local default) and sets the Java system property so the JNI library can be loaded.
- **GPU availability logging:** logs the detected GPU name and whether GPU phase correlation is active or if it falls back to CPU.

## CLIJX JAR modifications
- **New Java classes:** added `net.haesleinhuepf.clijx.fft.VkFFTPhaseCorrelation` and `net.haesleinhuepf.clijx.plugins.PhaseCorrelationFFT`.
- **JNI loader changes:** `VkFFTPhaseCorrelation` loads the native library `clijx_vkfft` and extracts OpenCL peer pointers from ClearCL/JOCL objects.
- **API surface:** `PhaseCorrelationFFT.phaseCorrelationShift(...)` exposes the GPU phase correlation to scripts (Jython/Groovy) and returns the shift vector.

## VkFFT implementation
- **Native library:** added `clijx_vkfft` built from `clijx/src/main/native/vkfft_phase_correlation/` with CMake.
- **JNI bridge:** `vkfft_phase_correlation.cpp` receives OpenCL context/queue/buffer pointers and runs VkFFT-based phase correlation on the GPU.
- **External dependency:** VkFFT sources are vendored under `clijx/third_party/VkFFT`.

## Behavior preserved
- Same dialog options and defaults (channel, ROI handling, max shifts, multi-time-scale, subpixel, virtual stack, only-compute mode).
- Same drift computation logic and multi-time-scale workflow.
- Same output behavior (corrected image or saved drift vectors).

## Additional runtime dependency
- Requires CLIJ2 and the CLIJX VkFFT phase correlation native library (`clijx_vkfft`) when using GPU phase correlation.
