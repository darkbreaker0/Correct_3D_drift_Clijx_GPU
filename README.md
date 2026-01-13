# Correct 2D/3D Drift (Fiji Jython) with GPU phase correlation (CLIJ2/VkFFT)

This repository contains a Fiji/ImageJ Jython script that measures and corrects translational drift in 2D or 3D time-lapse data. It uses phase correlation to estimate frame-to-frame shifts and can optionally apply sub-pixel correction.

GPU phase correlation (CLIJ2/VkFFT) is implemented via CLIJ2 and VkFFT:
- Convert frames to `ClearCLBuffer` objects with CLIJ2.
- Call a JNI wrapper (`VkFFTPhaseCorrelation`) that loads the native library `clijx_vkfft`.
- The JNI code receives the OpenCL context/queue/buffer pointers from ClearCL and runs VkFFT-based phase correlation on the GPU.
- A plugin wrapper (`PhaseCorrelationFFT`) exposes this to scripting and returns the shift vector.

This requires a compiled `clijx_vkfft` native library and a `CLIJX_VKFFT_PATH` (or PATH) entry so Fiji can load it.

## What it does
- Computes translation-only drift across time frames in 2D or 3D stacks
- Supports multi-time-scale drift estimation for slow drifts
- Optional sub-pixel correction using imglib2 interpolation
- Can limit analysis to a ROI that moves with the tracked structure
- Optional edge enhancement and background thresholding
- Can save output as a virtual stack to reduce RAM usage

## Getting started
1. Install the modified `clijx_-0.32.2.0.jar` into `Fiji.app/plugins/`.
2. Place `clijx_vkfft.dll` in `Fiji.app\lib\win64\` (auto-detected), or anywhere and set `CLIJX_VKFFT_PATH` to its full path.
3. Install `Correct_3D_drift_clij2.py` into your Plugins menu.
4. Restart Fiji and run the CLIJ2 script. The Log should report the GPU name and "GPU phase correlation active".

## Troubleshooting
- "GPU phase correlation unavailable, falling back to CPU": check `CLIJX_VKFFT_PATH` points to `clijx_vkfft.dll` and restart Fiji.
- "no clijx_vkfft in java.library.path": the native DLL is not found; set `CLIJX_VKFFT_PATH` or put the DLL on `PATH`.
- "Unable to access native OpenCL pointer": verify you are using the modified `clijx_-0.32.2.0.jar`, not the stock CLIJx JAR.

## Files
- `Correct_3D_drift_clij2.py`: CLIJ2/VkFFT-enabled variant
- `clijx_-0.32.2.0.jar`: modified CLIJx plugin with VkFFT phase correlation
- `clijx_vkfft.dll`: native VkFFT library for GPU phase correlation
- `installation.md`: how to install and register the script in Fiji
- `user_manual.md`: usage guide and option descriptions
- `change.md`: differences between the original and CLIJ2/VkFFT variants
- `LICENSES.md`: third-party license references for bundled code

## Links
- Original Correct_3D_Drift: https://github.com/fiji/Correct_3D_Drift
- CLIJx: https://github.com/clij/clijx
- VkFFT: https://github.com/DTolm/VkFFT

## License
This script is licensed under GPL-3.0 (see header in `Correct_3D_drift_clij2.py`).
