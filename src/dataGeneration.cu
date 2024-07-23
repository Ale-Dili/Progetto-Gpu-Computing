#include <stdio.h>
#include <stdlib.h>
#include "../../GPUcomputing/utils/common.h"
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <float.h>

//coordinata x e y massima [-10000 - radius, 10000 + radius]
#define MAX_X 10000
#define MAX_Y 10000


// Struttura per rappresentare un punto
typedef struct Points {
    float x;
    float y;
} Point;



__global__ void generatePointsGPU(Point* points, Point* roots, int nPoints, int nRoots, float radius, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nPoints) {

        curandState state;
        curand_init(seed, idx, 0, &state);

        //Seleziona casualmente un punto radice
        int root_idx = curand(&state) % nRoots;
        Point root = roots[root_idx];

        //Genera un angolo casuale
        float angle = curand_uniform(&state) * 2.0f * M_PI;

        //Genera una distanza casuale dal punto radice usando una distribuzione normale
        float distance = curand_normal(&state) * (radius/3 );


        //Calcola la posizione del nuovo punto
        points[idx].x = root.x + distance * cosf(angle);
        points[idx].y = root.y + distance * sinf(angle);
    }
}

void generatePointsCPU(Point* points, Point* roots, int nPoints, int nRoots, float radius) {
    for (int i = 0; i < nPoints; i++) {
        //Seleziona casualmente un punto radice
        int root_idx = rand() % nRoots;
        Point root = roots[root_idx];

        //Genera un angolo casuale
        float u1 = (float)rand() / (float)RAND_MAX;
        float angle = u1 * 2.0f * M_PI;

        //Genera una distanza casuale dal punto radice usando una distribuzione normale
        //BOX-MULLER
        float u2 = (float)rand() / (float)RAND_MAX;
        float u3 = (float)rand() / (float)RAND_MAX;
        float z = sqrt(-2.0f * log(u2)) * cos(2.0f * M_PI * u3);
        float distance = fabs(z) * (radius / 3.0f);


        points[i].x = root.x + distance * cosf(angle);
        points[i].y = root.y + distance * sinf(angle);
    }
}

void generateRandomCentroids(Point *centroids, int k) {
     // Inizializza il generatore di numeri casuali
    for (int i = 0; i < k; i++) {
        centroids[i].x = ((float)rand() / RAND_MAX) * 2 * MAX_X - MAX_X;
        centroids[i].y = ((float)rand() / RAND_MAX) * 2 * MAX_Y - MAX_Y;
    }
}


void writePoints(Point* points,int nPoints,const char* path) {
    FILE *file;
    file = fopen(path, "w");

    if (file) {
        printf("Stampo file\n");
        for (int i = 0; i < nPoints; ++i) {
            fprintf(file, "%f,%f\n", points[i].x, points[i].y);
        }
        fclose(file);
    } else {
        printf("Errore nell'apertura del file\n");
    }
}


int main() {

    double start, end;

    const int nPoints = 1000*1000; // Numero di punti da generare
    const int nRoots = 20; // Numero di punti radice
    const float radius = 1400.0f; // Raggio di distribuzione intorno ai punti radice

    //unsigned long int seed = time(NULL);
    unsigned long int seed = 42;
    srand(seed);

    //Alloco memoria per i punti e i punti radice
    Point* d_points;
    Point* d_roots;
    Point h_points[nPoints];
    cudaMalloc(&d_points, nPoints * sizeof(Point));
    cudaMalloc(&d_roots, nRoots * sizeof(Point));



    //h_roots viene usata sia per GPU che per CPU
    Point h_roots[nRoots];


    //Array per i punti CPU
    Point* points = (Point* ) malloc(nPoints * sizeof(Point));


    //Genera dei centroidi che vengono usati come root per i cluster
    generateRandomCentroids(h_roots, nRoots);

    /***********************************************************/
	/*                     CPU generation                      */
	/***********************************************************/
    printf("****************************\n");
    printf("CPU generation\n");
    printf("****************************\n");

    srand(seed);
    start = seconds();
    generatePointsCPU(points, h_roots, nPoints, nRoots, radius);
    end = seconds();

    double CPUtime = end - start;
    printf("Tempo di esecuzione CPU: %f\n", CPUtime);
    writePoints(points, nPoints, "src/PRJ/points_CPU.csv");

    free(points);

    printf("\n\n");
    /***********************************************************/
	/*                     GPU generation                      */
	/***********************************************************/
    printf("****************************\n");
    printf("GPU generation\n");
    printf("****************************\n");
    //Root da host a device
    cudaMemcpy(d_roots, h_roots, nRoots * sizeof(Point), cudaMemcpyHostToDevice);
    //Config il kernel
    int blockSize = 64;
    int numBlocks = (nPoints + blockSize - 1) / blockSize;

    start = seconds();
    generatePointsGPU<<<numBlocks, blockSize>>>(d_points, d_roots, nPoints, nRoots, radius, seed);
    cudaDeviceSynchronize();
    end = seconds();

    double GPUtime = end - start;
    printf("Tempo di esecuzione GPU: %f\n",GPUtime);
	double speedup = CPUtime/GPUtime;
	printf("    Speedup %.1f\n", speedup);

    //Risultato da device a host

    cudaMemcpy(h_points, d_points, nPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    writePoints(h_points, nPoints, "src/PRJ/points_GPU.csv");


    cudaFree(d_points);
    cudaFree(d_roots);

    return 0;
}
