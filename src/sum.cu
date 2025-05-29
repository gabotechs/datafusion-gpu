extern "C" __global__ void sum(float* input, float* result, int size) {
    // Shared memory for storing partial sums
    extern __shared__ float sharedData[];

    // Thread index within the block
    int threadId = threadIdx.x;
    // Global index in the array
    int globalIndex = blockIdx.x * blockDim.x + threadId;

    // Load elements into shared memory
    if (globalIndex < size) {
        sharedData[threadId] = input[globalIndex];
    } else {
        sharedData[threadId] = 0.0f;  // Handle out-of-bounds threads
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            sharedData[threadId] += sharedData[threadId + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (threadId == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}
