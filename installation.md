# Installation

## Requirements
- Fiji (ImageJ) with the standard Jython scripting support
- Fiji bundles the required libraries (mpicbg, imglib2, TransformJ)

## GPU/CLIJ2 requirements (optional)
- CLIJ2 and the modified CLIJx JAR (`clijx_-0.32.2.0.jar`) installed in `Fiji.app/plugins/`
- The native VkFFT library `clijx_vkfft.dll`
- Environment variable `CLIJX_VKFFT_PATH` pointing to the full path of `clijx_vkfft.dll`

## Install the script
Choose one of the following locations:

### Option A: Scripts menu (recommended)
1. Create a submenu folder if needed:
   `Fiji.app/scripts/Plugins/Registration/`
2. Copy `Correct_3D_drift.py` into that folder.
3. In Fiji: `Plugins > Scripting > Refresh Menus` (or restart Fiji).

Result: the script appears under `Plugins > Registration`.

### Option B: Plugins folder with manual registration
1. Copy `Correct_3D_drift.py` to `Fiji.app/plugins/`.
2. Edit `Fiji.app/plugins/plugins.config` and add:
   `Plugins, "Registration", "Correct 3D Drift", "Correct_3D_drift.py"`
3. Refresh menus or restart Fiji.

## Verify
Open a time-lapse image and run the script. You should see a dialog titled "Correct 2D/3D Drift Options".

## GPU/CLIJ2 setup (optional)
1. Copy `Correct_3D_drift_clij2.py` to the same menu location you used above.
2. Place `clijx_vkfft.dll` in `Fiji.app\lib\win64\` (auto-detected), or anywhere on disk.
3. If you do not use `Fiji.app\lib\win64\`, set the environment variable:
   `CLIJX_VKFFT_PATH=C:\Apps\clijx_vkfft.dll`
4. Restart Fiji and run the CLIJ2 script. The Log window should report the GPU name and "GPU phase correlation active".

## Developer/integration notes
- Native VkFFT sources live under `clijx-local/src/main/native/vkfft_phase_correlation/`.
- VkFFT is vendored under `clijx-local/third_party/VkFFT` and is required to rebuild `clijx_vkfft.dll`.
