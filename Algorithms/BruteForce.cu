#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <chrono>

__device__ void generatePermutation(int n, int threadId, char* permutation) {
    // Generate permutation using thread id and n
    for (int i = 0; i < n; i++) {
        permutation[i] = (threadId + i) % n;
    }
}

__device__ void evaluatePermutation(int n, char* permutation, float** graph, int* evaluation, int threadId) {
    // Evaluate permutation
    *evaluation = threadId;
}

__global__ void generateAndEvaluatePermutations(const int n, const unsigned long long int nPermutations,
                                                float** graph, int* bestEvaluation, char* bestPermutation) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < nPermutations) {
        char* permutation = (char*)malloc(n * sizeof(char));
        int* evaluation = (int*)malloc(sizeof(int));

        generatePermutation(n, threadId, permutation);
        evaluatePermutation(n, permutation, graph, evaluation, threadId);

        assert(*evaluation == threadId);
        if (*evaluation > *bestEvaluation) {
            
            atomicMax(bestEvaluation, *evaluation);
            *bestPermutation = *permutation;
        }

        free(permutation);
        free(evaluation);
    }
}

char* BruteForce(const int n, int d, float** graph) {
    //
    const unsigned long long int n_permutations = tgamma(n+d+1);

    char* bestPermutation = (char*)malloc(sizeof(char) * (n + d));
    char* threadPermutation;
    int* threadEvaluation;

    constexpr float initialEvaluation = 0;

    cudaMalloc(&threadPermutation, sizeof(char) * (n + d)); // Should hold a route permutation
    cudaMalloc(&threadEvaluation, sizeof(int));

    cudaMemcpy(threadEvaluation, &initialEvaluation, sizeof(float), cudaMemcpyHostToDevice);

    constexpr int blockSize = 256;
    int numBlocks = ceil(n_permutations / (float)blockSize);

    generateAndEvaluatePermutations<<<numBlocks, blockSize>>>(n, n_permutations, graph,
                                                              threadEvaluation, threadPermutation);

    cudaMemcpy(bestPermutation, threadPermutation, sizeof(char), cudaMemcpyDeviceToHost);

    std::cout << "Best permutation: ";
    for (int i = 0; i < n; i++) {
        std::cout << bestPermutation[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(threadPermutation);
    cudaFree(threadEvaluation);

    return bestPermutation;
}

int main() {
    int n = 3;
    int d = 1;

    char* route = (char*)malloc(sizeof(char) * (n+d));

    // adjacency matrix
    float init[4][4] = {
        {0, 1, 1, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 0}
    };

    float** graph = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        graph[i] = init[i];
    }

    route = BruteForce(n, d, graph);

    for (int i = 0; i < (n + d); i++) {
        std::cout << route[i] << " ";
    }
    std::cout << std::endl;

    free(graph);
    free(route);
}

/*
int main()
{
    int n = 50;

    auto serial_begin = std::chrono::high_resolution_clock::now();
    unsigned long long int serial_result = tgamma(n+1);
    auto serial_end = std::chrono::high_resolution_clock::now();

    std::chrono::high_resolution_clock::time_point parallel_begin = std::chrono::high_resolution_clock::now();
    std::vector<int> n_vector(n-1);
    std::iota(n_vector.begin(), n_vector.end(), 2);
    unsigned long long int parallel_result = thrust::reduce(thrust::host, n_vector.begin(), n_vector.end(),
        1, thrust::multiplies<int>());
    std::chrono::high_resolution_clock::time_point parallel_end = std::chrono::high_resolution_clock::now();

    auto serial_time_taken = serial_end - serial_begin;
    auto parallel_time_taken = parallel_end - parallel_begin;

    std::cout << "Serial result: " << serial_result << std::endl;
    std::cout << "Parallel result: " << parallel_result << std::endl << std::endl;

    std::cout << "Serial time: " << std::chrono::duration<double>(serial_time_taken).count() << std::endl;
    std::cout << "Parallel time: " << std::chrono::duration<double>(parallel_time_taken).count() << std::endl;

    return 0;
}*/