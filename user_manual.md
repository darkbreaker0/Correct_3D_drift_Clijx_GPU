# User Manual

## Overview
The script computes translational drift between time frames using phase correlation and applies the inverse shift to stabilize the time series. It works with 2D or 3D data and can optionally correct sub-pixel drift.

## How to run
1. Open a time-lapse stack or hyperstack in Fiji.
2. Run `Correct_3D_drift.py` from the Plugins menu.
3. Configure options and click OK.

## Options in the dialog
- Channel for registration: which channel is used to compute drift (applied to all channels).
- Correct only x & y (for 3D data): ignores z drift and only corrects XY.
- Multi time scale computation: estimates drift at larger time gaps to capture slow drifts.
- Sub pixel drift correction: applies sub-pixel translation using imglib2 interpolation.
- Edge enhance images: performs light smoothing and edge detection to improve correlation.
- Only consider pixels with values larger than: subtracts background threshold before correlation.
- Lowest z plane / Highest z plane: limit the z range used for drift computation.
- Max shift x/y/z: limits the allowed drift per time step in pixels.
- Use virtualstack for saving results to disk: writes corrected frames to a folder to save RAM.
- Only compute drift vectors: do not apply correction; save the shift table instead.

## ROI tracking
If a ROI is present on the image, drift is computed only inside that ROI. The ROI is moved along with the drift so it follows the same structure over time.

## Outputs
- Corrected image: shown as a new ImagePlus.
- If "Only compute drift vectors" is enabled: a text file containing dx/dy/dz per frame and the ROI bounds.
- If virtual stack output is enabled: individual corrected frames are saved as TIFFs in the selected folder.

## Tips
- For noisy data, try edge enhancement and set a background threshold.
- If drift is slow and sub-pixel, enable multi-time-scale and sub-pixel correction.
- Use max shift limits to prevent outlier frames from causing large jumps.