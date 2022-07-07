#include <iostream>

__global__ void kernel(
		std::uint32_t* const ptr,
		std::uint32_t* const src_ptr
		) {
	ptr[threadIdx.x] = atomicAdd(src_ptr, 1u);
}

void test(
		const std::uint32_t num_threads
		) {
	std::uint32_t* src_ptr;
	cudaMallocManaged(&src_ptr, sizeof(std::uint32_t));
	cudaMemset(src_ptr, 0, sizeof(std::uint32_t));

	std::uint32_t* result_ptr;
	cudaMallocManaged(&result_ptr,   sizeof(std::uint32_t) * num_threads);
	cudaMemset(result_ptr, 0, sizeof(std::uint32_t) * num_threads);

	kernel<<<1, num_threads>>>(
			result_ptr,
			src_ptr
			);

	cudaDeviceSynchronize();

	for (std::uint32_t i = 0; i < num_threads; i++) {
		std::printf("[%4u] : %u\n", i, result_ptr[i]);
	}
}

int main() {
	test(128);
}
