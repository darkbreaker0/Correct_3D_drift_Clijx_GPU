package net.haesleinhuepf.clijx.fft;

import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.clearcl.ClearCLContext;
import net.haesleinhuepf.clij.clearcl.ClearCLQueue;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;

/**
 * JNI wrapper around VkFFT-based phase correlation.
 * Requires a native library built from src/main/native/vkfft_phase_correlation.
 */
public class VkFFTPhaseCorrelation {

    static {
        String explicitPath = System.getProperty("CLIJX_VKFFT_PATH");
        if (explicitPath == null || explicitPath.length() == 0) {
            explicitPath = System.getenv("CLIJX_VKFFT_PATH");
        }
        if (explicitPath != null && explicitPath.length() > 0) {
            System.load(explicitPath);
        } else {
            System.loadLibrary("clijx_vkfft");
        }
    }

    private VkFFTPhaseCorrelation() {
        // utility
    }

    public static float[] computeShift(CLIJ2 clij2, ClearCLBuffer input1, ClearCLBuffer input2) {
        ClearCLBuffer src1 = input1;
        ClearCLBuffer src2 = input2;
        ClearCLBuffer tmp1 = null;
        ClearCLBuffer tmp2 = null;
        try {
            if (input1.getNativeType() != NativeTypeEnum.Float) {
                tmp1 = clij2.create(input1.getDimensions(), NativeTypeEnum.Float);
                clij2.copy(input1, tmp1);
                src1 = tmp1;
            }
            if (input2.getNativeType() != NativeTypeEnum.Float) {
                tmp2 = clij2.create(input2.getDimensions(), NativeTypeEnum.Float);
                clij2.copy(input2, tmp2);
                src2 = tmp2;
            }

            ClearCLContext context = clij2.getCLIJ().getClearCLContext();
            ClearCLQueue queue = context.getDefaultQueue();

            long contextPtr = getPeerPointer(context);
            long queuePtr = getPeerPointer(queue);
            long buf1Ptr = getPeerPointer(src1);
            long buf2Ptr = getPeerPointer(src2);

            int width = (int) src1.getWidth();
            int height = (int) src1.getHeight();
            int depth = (int) src1.getDepth();

            return computeShiftNative(contextPtr, queuePtr, buf1Ptr, buf2Ptr, width, height, depth);
        } finally {
            if (tmp1 != null) {
                tmp1.close();
            }
            if (tmp2 != null) {
                tmp2.close();
            }
        }
    }

    private static long getPeerPointer(Object clearCLObject) {
        Object current = clearCLObject;
        for (int i = 0; i < 4; i++) {
            if (current instanceof Number) {
                return ((Number) current).longValue();
            }
            try {
                Object peer = current.getClass().getMethod("getPeerPointer").invoke(current);
                if (peer != null && peer != current) {
                    current = peer;
                    continue;
                }
            } catch (Exception ignored) {
                // fall through
            }
            try {
                Object ptr = current.getClass().getMethod("getPointer").invoke(current);
                if (ptr != null && ptr != current) {
                    current = ptr;
                    continue;
                }
            } catch (Exception ignored) {
                // fall through
            }
            try {
                Object nativePtr = current.getClass().getMethod("getNativePointer").invoke(current);
                if (nativePtr instanceof Number) {
                    return ((Number) nativePtr).longValue();
                }
            } catch (Exception ignored) {
                // fall through
            }
            break;
        }
        throw new IllegalStateException("Unable to access native OpenCL pointer from " + clearCLObject.getClass().getName());
    }

    private static native float[] computeShiftNative(long contextPtr, long queuePtr, long input1Ptr, long input2Ptr,
                                                     int width, int height, int depth);
}
