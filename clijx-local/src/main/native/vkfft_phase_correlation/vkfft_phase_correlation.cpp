#include <jni.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#define VKFFT_BACKEND 3
#include "vkFFT.h"

static void throwJava(JNIEnv* env, const char* msg) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass) {
        env->ThrowNew(exClass, msg);
    }
}

static bool checkCL(JNIEnv* env, cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::string full = std::string(msg) + " (OpenCL error " + std::to_string(err) + ")";
        throwJava(env, full.c_str());
        return false;
    }
    return true;
}

static cl_device_id getDeviceFromContext(JNIEnv* env, cl_context context) {
    size_t size = 0;
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &size);
    if (!checkCL(env, err, "clGetContextInfo size failed")) return nullptr;
    if (size < sizeof(cl_device_id)) {
        throwJava(env, "No OpenCL device found in context");
        return nullptr;
    }
    cl_device_id device = nullptr;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, nullptr);
    if (!checkCL(env, err, "clGetContextInfo device failed")) return nullptr;
    return device;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_net_haesleinhuepf_clijx_fft_VkFFTPhaseCorrelation_computeShiftNative(
        JNIEnv* env, jclass, jlong contextPtr, jlong queuePtr,
        jlong input1Ptr, jlong input2Ptr, jint width, jint height, jint depth) {

    if (width <= 0 || height <= 0 || depth <= 0) {
        throwJava(env, "Invalid dimensions for phase correlation");
        return nullptr;
    }

    cl_context context = reinterpret_cast<cl_context>(contextPtr);
    cl_command_queue queue = reinterpret_cast<cl_command_queue>(queuePtr);
    cl_mem input1 = reinterpret_cast<cl_mem>(input1Ptr);
    cl_mem input2 = reinterpret_cast<cl_mem>(input2Ptr);

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(depth);
    const size_t bytesReal = n * sizeof(float);
    const size_t bytesComplex = n * sizeof(cl_float2);

    cl_int err = CL_SUCCESS;
    cl_mem buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, bytesComplex, nullptr, &err);
    if (!checkCL(env, err, "clCreateBuffer buf1 failed")) return nullptr;
    cl_mem buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, bytesComplex, nullptr, &err);
    if (!checkCL(env, err, "clCreateBuffer buf2 failed")) return nullptr;
    cl_mem bufCorr = clCreateBuffer(context, CL_MEM_READ_WRITE, bytesComplex, nullptr, &err);
    if (!checkCL(env, err, "clCreateBuffer bufCorr failed")) return nullptr;

    const char* src =
        "__kernel void pack_real_to_complex(__global const float* src, __global float2* dst, int n) {"
        "  int gid = get_global_id(0);"
        "  if (gid < n) {"
        "    dst[gid] = (float2)(src[gid], 0.0f);"
        "  }"
        "}"
        "__kernel void cross_power(__global const float2* a, __global const float2* b, __global float2* out, int n) {"
        "  int gid = get_global_id(0);"
        "  if (gid < n) {"
        "    float2 av = a[gid];"
        "    float2 bv = b[gid];"
        "    float2 conjb = (float2)(bv.x, -bv.y);"
        "    float2 prod = (float2)(av.x*conjb.x - av.y*conjb.y, av.x*conjb.y + av.y*conjb.x);"
        "    float mag = sqrt(prod.x*prod.x + prod.y*prod.y);"
        "    if (mag > 0.0f) {"
        "      prod.x /= mag;"
        "      prod.y /= mag;"
        "    }"
        "    out[gid] = prod;"
        "  }"
        "}";

    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
    if (!checkCL(env, err, "clCreateProgramWithSource failed")) return nullptr;

    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (!checkCL(env, err, "clBuildProgram failed")) return nullptr;

    cl_kernel packKernel = clCreateKernel(program, "pack_real_to_complex", &err);
    if (!checkCL(env, err, "clCreateKernel pack failed")) return nullptr;
    cl_kernel crossKernel = clCreateKernel(program, "cross_power", &err);
    if (!checkCL(env, err, "clCreateKernel cross failed")) return nullptr;

    // Pack inputs into complex buffers
    err = clSetKernelArg(packKernel, 0, sizeof(cl_mem), &input1);
    err |= clSetKernelArg(packKernel, 1, sizeof(cl_mem), &buf1);
    err |= clSetKernelArg(packKernel, 2, sizeof(int), &n);
    if (!checkCL(env, err, "clSetKernelArg pack1 failed")) return nullptr;
    size_t global = n;
    err = clEnqueueNDRangeKernel(queue, packKernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    if (!checkCL(env, err, "clEnqueueNDRangeKernel pack1 failed")) return nullptr;

    err = clSetKernelArg(packKernel, 0, sizeof(cl_mem), &input2);
    err |= clSetKernelArg(packKernel, 1, sizeof(cl_mem), &buf2);
    err |= clSetKernelArg(packKernel, 2, sizeof(int), &n);
    if (!checkCL(env, err, "clSetKernelArg pack2 failed")) return nullptr;
    err = clEnqueueNDRangeKernel(queue, packKernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    if (!checkCL(env, err, "clEnqueueNDRangeKernel pack2 failed")) return nullptr;

    clFinish(queue);

    // VkFFT setup
    cl_device_id device = getDeviceFromContext(env, context);
    if (!device) return nullptr;

    VkFFTConfiguration config = VKFFT_ZERO_INIT;
    config.FFTdim = (depth > 1) ? 3 : 2;
    config.size[0] = static_cast<pfUINT>(width);
    config.size[1] = static_cast<pfUINT>(height);
    config.size[2] = static_cast<pfUINT>(depth);
    config.device = &device;
    config.context = &context;
    config.commandQueue = &queue;
    config.buffer = &buf1;
    config.normalize = 1;

    VkFFTApplication app = VKFFT_ZERO_INIT;
    VkFFTResult res = initializeVkFFT(&app, config);
    if (res != VKFFT_SUCCESS) {
        throwJava(env, "initializeVkFFT failed");
        return nullptr;
    }

    VkFFTLaunchParams launchParams = VKFFT_ZERO_INIT;
    launchParams.commandQueue = &queue;

    launchParams.buffer = &buf1;
    res = VkFFTAppend(&app, 0, &launchParams);
    if (res != VKFFT_SUCCESS) {
        deleteVkFFT(&app);
        throwJava(env, "VkFFTAppend forward (input1) failed");
        return nullptr;
    }

    launchParams.buffer = &buf2;
    res = VkFFTAppend(&app, 0, &launchParams);
    if (res != VKFFT_SUCCESS) {
        deleteVkFFT(&app);
        throwJava(env, "VkFFTAppend forward (input2) failed");
        return nullptr;
    }

    // Cross power spectrum on GPU
    err = clSetKernelArg(crossKernel, 0, sizeof(cl_mem), &buf1);
    err |= clSetKernelArg(crossKernel, 1, sizeof(cl_mem), &buf2);
    err |= clSetKernelArg(crossKernel, 2, sizeof(cl_mem), &bufCorr);
    err |= clSetKernelArg(crossKernel, 3, sizeof(int), &n);
    if (!checkCL(env, err, "clSetKernelArg cross failed")) return nullptr;
    err = clEnqueueNDRangeKernel(queue, crossKernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    if (!checkCL(env, err, "clEnqueueNDRangeKernel cross failed")) return nullptr;

    clFinish(queue);

    // Inverse FFT of cross power spectrum
    launchParams.buffer = &bufCorr;
    res = VkFFTAppend(&app, 1, &launchParams);
    deleteVkFFT(&app);
    if (res != VKFFT_SUCCESS) {
        throwJava(env, "VkFFTAppend inverse failed");
        return nullptr;
    }

    clFinish(queue);

    // Read back correlation and find peak
    std::vector<cl_float2> hostCorr(n);
    err = clEnqueueReadBuffer(queue, bufCorr, CL_TRUE, 0, bytesComplex, hostCorr.data(), 0, nullptr, nullptr);
    if (!checkCL(env, err, "clEnqueueReadBuffer failed")) return nullptr;

    size_t maxIdx = 0;
    float maxVal = hostCorr[0].s[0];
    for (size_t i = 1; i < n; ++i) {
        float v = hostCorr[i].s[0];
        if (v > maxVal) {
            maxVal = v;
            maxIdx = i;
        }
    }

    const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);
    int z = static_cast<int>(maxIdx / plane);
    size_t rem = maxIdx % plane;
    int y = static_cast<int>(rem / width);
    int x = static_cast<int>(rem % width);

    int shiftX = (x <= width / 2) ? x : x - width;
    int shiftY = (y <= height / 2) ? y : y - height;
    int shiftZ = (depth <= 1) ? 0 : ((z <= depth / 2) ? z : z - depth);

    clReleaseKernel(packKernel);
    clReleaseKernel(crossKernel);
    clReleaseProgram(program);
    clReleaseMemObject(buf1);
    clReleaseMemObject(buf2);
    clReleaseMemObject(bufCorr);

    jfloatArray out = env->NewFloatArray(3);
    if (!out) {
        throwJava(env, "Failed to allocate output array");
        return nullptr;
    }
    jfloat vals[3];
    vals[0] = static_cast<jfloat>(shiftX);
    vals[1] = static_cast<jfloat>(shiftY);
    vals[2] = static_cast<jfloat>(shiftZ);
    env->SetFloatArrayRegion(out, 0, 3, vals);
    return out;
}
