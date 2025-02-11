#include<cuda_runtime.h>
#include<limits>
#include<cmath>
#include<omp.h>

extern "C" {
    __constant__ int c_graph[625]; // 25x25 graph (max returned by openStreetMaps)
    __constant__ char c_numLocations, c_numDays, c_routeLength; // Values frequently accessed that do not change

    __device__ void generateRoute(char* baseSet, unsigned long long routeNumber, const unsigned long long nRoutes,
        int localThreadIndex, char* route, char* routes) {
        unsigned long long nFactorial = nRoutes / (c_routeLength - 1);

        for (char i = c_routeLength - 2; i >= 0; i--) {
            char selectedIndex = routeNumber / nFactorial;
            route[i] = baseSet[selectedIndex];
            baseSet[selectedIndex] = baseSet[i]; // Reorganizes array so chosen option can't be selected again

            if (i > 0) {
                routeNumber %= nFactorial;
                nFactorial /= i;
            }
        }

        route[c_routeLength - 1] = 0; // All routes end going back to centre
        for (int i = 0; i < c_routeLength; i++) routes[(localThreadIndex * c_routeLength) + i] = route[i];
    }

    // Should place this method else where, for use across multiple algorithms
    __device__ void evaluateRoute(char* route, int localThreadIndex, float* evaluations) {
        float evaluation = 0;
        char index;
        char previousIndex = 0;
        char currentDay = 0;
        float* evaluationPerDay = new float[c_numDays];

        // Sum time taken for route
        for (int i = 0; i < c_routeLength; i++) {
            index = route[i];
            if (previousIndex == index || // Invalid route (visited centre twice in row, or first)
                currentDay >= c_numDays) // Too many days in route (not sure if this case is possible?)
            {
                evaluations[localThreadIndex] = FLT_MAX;
                delete[] evaluationPerDay;
                return;
            }

            evaluation += c_graph[(previousIndex * c_numLocations) + index];
            evaluationPerDay[currentDay] += c_graph[(previousIndex * c_numLocations) + index];

            if (index == 0) currentDay++;
            previousIndex = index;
        }

        // Calculate standard deviation of route length between days
        float mean = evaluation / c_numDays;
        float meanDeviation = 0.0;
        for (int i = 0; i < c_numDays; i++) {
            float differenceFromMean = mean - evaluationPerDay[i];
            meanDeviation += differenceFromMean * differenceFromMean;
        }
        float variance = meanDeviation / c_numDays;
        float standardDeviation = sqrt(variance);

        // Multiply evaluation by standard deviation.
        evaluation *= 1 + standardDeviation;

        // Save evaluation to shared block memory
        evaluations[localThreadIndex] = evaluation;
        delete[] evaluationPerDay;
    }

    __device__ void getBestRouteDevice(int* indexes, float* evaluations, int localThreadIndex, int globalThreadIndex) {
        indexes[localThreadIndex] = globalThreadIndex;
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            __syncthreads();
            if (localThreadIndex % (2 * stride) != 0) continue;
            if (evaluations[globalThreadIndex] <= evaluations[globalThreadIndex + stride]) continue;

            evaluations[globalThreadIndex] = evaluations[globalThreadIndex + stride];
            indexes[localThreadIndex] = indexes[localThreadIndex + stride]; // To keep track of the best route
        }
    }

    __global__ void getBestRouteGlobal(int nRoutes, bool firstBlock, float* d_evaluations, char* d_routes) {
        extern __shared__ int sharedMemory[];
        int blockSize = blockDim.x;
        float* evaluations = (float*)sharedMemory;
        int* indexes = (int*)&sharedMemory[blockSize];

        int localThreadIndex = threadIdx.x;
        int blockIdxOffset = firstBlock ? 0 : 1;
        int globalThreadIndex = ((blockIdx.x + blockIdxOffset) * blockDim.x) + localThreadIndex;

        if (globalThreadIndex < nRoutes) {
            getBestRouteDevice(indexes, evaluations, localThreadIndex, globalThreadIndex);
        }
        __syncthreads(); // Unsure if needed

        int blockIndex = blockIdx.x;

        if (localThreadIndex == 0) d_evaluations[blockIndex] = evaluations[0];

        if (localThreadIndex < c_routeLength) {
            int bestIndex = indexes[0];

            char* destinationPointer = &d_routes[blockIndex * c_routeLength];
            char* sourcePointer = &d_routes[(blockIndex + bestIndex) * c_routeLength];

            destinationPointer[localThreadIndex] = sourcePointer[localThreadIndex];
        }
    }

    // Call like <<<blocks, blocksize, blocksize*sizeof(int) + blocksize*routeLength*sizeof(char)>>>
    __global__ void generateAndEvaluateRoutes(const unsigned long long nRoutes, const unsigned long long routesEvaluated,
        float* d_evaluations, char* d_routes) {
        extern __shared__ int sharedMemory[];
        int blockSize = blockDim.x;
        float* evaluations = (float*)sharedMemory;
        int* indexes = (int*)&sharedMemory[blockSize];
        char* routes = (char*)&sharedMemory[2 * blockSize];

        int blockIndex = blockIdx.x;
        int localThreadIndex = threadIdx.x;
        int globalThreadIndex = blockIndex * blockSize + localThreadIndex;

        if (routesEvaluated + globalThreadIndex < nRoutes) {
            unsigned long long routeNumber = routesEvaluated + globalThreadIndex;

            char* route = new char[c_routeLength];
            char* baseSet = new char[c_routeLength - 1];

            for (int i = 0; i < c_numDays - 1; i++) baseSet[i] = 0;
            for (int i = 0; i < c_numLocations - 1; i++) baseSet[i + c_numDays - 1] = i + 1;

            generateRoute(baseSet, routeNumber, nRoutes, localThreadIndex, route, routes);
            evaluateRoute(route, localThreadIndex, evaluations);

            delete[] baseSet;
            delete[] route;
        }
        else {
            evaluations[localThreadIndex] = FLT_MAX; // Ensures when getting best eval, doesn't pick null route
        }

        getBestRouteDevice(indexes, evaluations, localThreadIndex, globalThreadIndex);
        __syncthreads(); // Unsure if needed

        if (localThreadIndex == 0) d_evaluations[blockIndex] = evaluations[0]; // Set d_evaluations[blockIndex] to best evaluation

        if (localThreadIndex < c_routeLength) {
            int bestIndex = indexes[0];

            char* destinationPointer = &d_routes[blockIndex * c_routeLength];
            char* sourcePointer = &routes[bestIndex * c_routeLength];

            destinationPointer[localThreadIndex] = sourcePointer[localThreadIndex];
        }
    }

    __declspec(dllexport) char* bruteForce(char n, char d, int* graph) {
        constexpr unsigned int maxRoutesStored = 1048576; // 1024 * 1024;

        // Number of locations + times returning to centre (minus 1 because centre included in n)
        const char routeLength = n + d - 1;

        // Should store up to 20! (even if calculations take eternity)
        unsigned long long nRoutes = 1;
#pragma omp parallel for reduction(*:nRoutes)
        for (int i = 2; i < routeLength; i++) nRoutes *= i; // this is (routeLength - 1)! we do -1 because all routes end going back to 0.

        unsigned long long routesEvaluated = 0;

        float* d_evaluations;
        char* d_routes;

        cudaMalloc(&d_evaluations, maxRoutesStored * sizeof(float));
        cudaMalloc(&d_routes, maxRoutesStored * sizeof(char));
        cudaMemcpyToSymbol(c_graph, graph, n * n * sizeof(int));
        cudaMemcpyToSymbol(c_numLocations, &n, sizeof(char));
        cudaMemcpyToSymbol(c_numDays, &d, sizeof(char));
        cudaMemcpyToSymbol(c_routeLength, &routeLength, sizeof(char));

        const int blockSize = 256; // Update this to calculate based on max occupancy

        int sharedMemorySize = blockSize * sizeof(float) + blockSize * sizeof(int) + blockSize * routeLength * sizeof(char);
        char* bestRoute = new char[routeLength];
        float bestEvaluation = FLT_MAX;

        while (routesEvaluated < nRoutes) {
            const unsigned int numBlocks = std::min<int>(((nRoutes - routesEvaluated) + blockSize - 1) / blockSize, maxRoutesStored);

            generateAndEvaluateRoutes << <numBlocks, blockSize, sharedMemorySize >> > (nRoutes, routesEvaluated, d_evaluations, d_routes);

            int routesStored = numBlocks;
            while (routesStored > 1) {
                int secondSharedMemorySize = blockSize * sizeof(float) + blockSize * sizeof(int);
                int secondNumBlocks = (routesStored + blockSize - 1) / blockSize; // Integer division rounded up

                getBestRouteGlobal << <1, blockSize, secondSharedMemorySize >> > (blockSize, true, d_evaluations, d_routes); // This is goofy, but need first block to finish first
                if (secondNumBlocks > 1) {
                    getBestRouteGlobal << <secondNumBlocks - 1, blockSize, secondSharedMemorySize >> > (routesStored, false, d_evaluations, d_routes);
                }

                routesStored = secondNumBlocks;
            }

            cudaMemcpy(bestRoute, d_routes, routeLength * sizeof(char), cudaMemcpyDeviceToHost);
            cudaMemcpy(&bestEvaluation, d_evaluations, sizeof(float), cudaMemcpyDeviceToHost);

            routesEvaluated += numBlocks * blockSize;
        }

        cudaFree(d_routes);
        cudaFree(d_evaluations);

        return bestRoute;
    }
}