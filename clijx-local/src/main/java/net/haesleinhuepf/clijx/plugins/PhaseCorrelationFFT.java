package net.haesleinhuepf.clijx.plugins;

import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clijx.fft.VkFFTPhaseCorrelation;

/**
 * GPU phase correlation using VkFFT (native library required).
 */
public class PhaseCorrelationFFT {

    private PhaseCorrelationFFT() {
        // utility
    }

    public static float[] phaseCorrelationShift(CLIJ2 clij2, ClearCLBuffer input1, ClearCLBuffer input2) {
        return VkFFTPhaseCorrelation.computeShift(clij2, input1, input2);
    }
}
