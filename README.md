# Correct 2D/3D Drift (Fiji Jython)

This repository contains a Fiji/ImageJ Jython script that measures and corrects translational drift in 2D or 3D time-lapse data. It uses phase correlation to estimate frame-to-frame shifts and can optionally apply sub-pixel correction.

## What it does
- Computes translation-only drift across time frames in 2D or 3D stacks
- Supports multi-time-scale drift estimation for slow drifts
- Optional sub-pixel correction using imglib2 interpolation
- Can limit analysis to a ROI that moves with the tracked structure
- Optional edge enhancement and background thresholding
- Can save output as a virtual stack to reduce RAM usage

## Quick start
1. Open a time-lapse stack or hyperstack in Fiji.
2. Run the script from the Plugins menu (see INSTALLATION.md).
3. Choose options in the dialog and run.

## GPU phase correlation (CLIJ2/VkFFT)
In the GPU-accelerated variant (implemented in the related CLIJX codebase), phase correlation is computed on the GPU using VkFFT. The flow is:
- Convert frames to `ClearCLBuffer` objects with CLIJ2.
- Call a JNI wrapper (`VkFFTPhaseCorrelation`) that loads the native library `clijx_vkfft`.
- The JNI code receives the OpenCL context/queue/buffer pointers from ClearCL and runs VkFFT-based phase correlation on the GPU.
- A plugin wrapper (`PhaseCorrelationFFT`) exposes this to scripting and returns the shift vector.

This requires a compiled `clijx_vkfft` native library and a `CLIJX_VKFFT_PATH` (or PATH) entry so Fiji can load it.

## Files
- `Correct_3D_drift.py`: main Jython script
- `Correct_3D_drift_clij2.py`: CLIJ2/VkFFT-enabled variant
- `installation.md`: how to install and register the script in Fiji
- `user_manual.md`: usage guide and option descriptions
- `change.md`: differences between the original and CLIJ2/VkFFT variants
- `LICENSES.md`: third-party license references for bundled code

## Links
- Original Correct_3D_Drift: https://github.com/fiji/Correct_3D_Drift
- CLIJx: https://github.com/clij/clijx
- VkFFT: https://github.com/DTolm/VkFFT

## License
This script is licensed under GPL-3.0 (see header in `Correct_3D_drift.py`).
