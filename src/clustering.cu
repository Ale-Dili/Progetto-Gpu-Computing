#include <stdio.h>
#include <stdlib.h>
#include "../../GPUcomputing/utils/common.h"
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <float.h>

//coordinata x e y massima [-1000 - radius, 1000 + radius]
#define MAX_X 10000
#define MAX_Y 10000


typedef struct {
    float x;
    float y;
} Point;

void readPoints(const char* path, Point* points) {
    FILE *file;
    file = fopen(path, "r");
    if (file) {
      char line[256];
      float x, y;
      int i = 0;
      while (fgets(line, sizeof(line), file)) {
        // Parse the line to extract two floats
        if (sscanf(line, "%f,%f", &x, &y) == 2) {
            // Process the extracted numbers
            points[i].x = x;
            points[i].y = y;
        } else {
            fprintf(stderr, "Error parsing line: %s", line);
        }
        i++;
      }

    }else{
        printf("Errore nell'apertura del file\n");
    }
}


void generateRandomCentroids(Point *centroids, int k) {
    srand(time(NULL)); // Inizializza il generatore di numeri casuali
    for (int i = 0; i < k; i++) {
        centroids[i].x = ((float)rand() / RAND_MAX) * 2 * MAX_X - MAX_X;
        centroids[i].y = ((float)rand() / RAND_MAX) * 2 * MAX_Y - MAX_Y;
    }
}

void deepCopyPoints(Point* dest, Point* src, int nCluster) {
    for (int i = 0; i < nCluster; ++i) {
        dest[i] = src[i];

    }
}


float distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void kmeansCPU(Point *points, int num_points, Point *centroids, int num_centroids, int *labels) {
    int changed;
    Point *new_centroids = (Point *)malloc(num_centroids * sizeof(Point));
    int *counts = (int *)malloc(num_centroids * sizeof(int));

    do {
        //Reset dei centroidi
        for (int i = 0; i < num_centroids; i++) {
            new_centroids[i].x = 0;
            new_centroids[i].y = 0;
            counts[i] = 0;
        }

        changed = 0;

        //Assegno ad ogni punto il centroide più vicino
        for (int i = 0; i < num_points; i++) {
            float min_dist = FLT_MAX;
            int closest_centroid = 0;
            for (int j = 0; j < num_centroids; j++) {
                float dist = distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            //Stopping criteria
            if (labels[i] != closest_centroid) {
                labels[i] = closest_centroid;
                changed = 1;
            }
            new_centroids[closest_centroid].x += points[i].x;
            new_centroids[closest_centroid].y += points[i].y;
            counts[closest_centroid]++;
        }

        // Recalculate centroids
        for (int i = 0; i < num_centroids; i++) {
            if (counts[i] != 0) {
                centroids[i].x = new_centroids[i].x / counts[i];
                centroids[i].y = new_centroids[i].y / counts[i];
            }
        }

    } while (changed);

    free(new_centroids);
    free(counts);
}

void writePointsV2(Point *points,int* labels, int nPoints,const char* path) {
    FILE *file;
    file = fopen(path, "w");

    if (file) {
        printf("Stampo file\n");
        for (int i = 0; i < nPoints; ++i) {
            fprintf(file, "%f,%f,%d\n", points[i].x, points[i].y, labels[i]);
        }
        fclose(file);
    } else {
        printf("Errore nell'apertura del file\n");
    }
}

__global__ void kmeansSumPoints(Point* points, Point* centroids, int* labels, int* count, Point* sum_points, int nPoints, int n_centroids, int* d_changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nPoints) {
        int closestCentroid = -1;
        float minDistance = FLT_MAX;
        float distance = 0.0f;
        for (int i = 0; i < n_centroids; i++) {
            distance = sqrtf((points[idx].x - centroids[i].x) * (points[idx].x - centroids[i].x) + (points[idx].y - centroids[i].y) * (points[idx].y - centroids[i].y));
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = i;
            }
        }
        if (labels[idx] != closestCentroid) {
            labels[idx] = closestCentroid;
            atomicAdd(d_changed, 1);
        }
        atomicAdd(&sum_points[closestCentroid].x, points[idx].x);
        atomicAdd(&sum_points[closestCentroid].y, points[idx].y);
        atomicAdd(&count[closestCentroid], 1);
    }
}

__global__ void kmeansUpdateCentroids(Point* centroids, Point* sum_points, int* count, int n_centroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_centroids) {
        if (count[idx] > 0) {
            centroids[idx].x = sum_points[idx].x / count[idx];
            centroids[idx].y = sum_points[idx].y / count[idx];
        }
    }
}

int main() {
    double start, end;

    double endNoTransf, endTransf;

    const int nCluster = 12;

    const int nPoints = 1000*1000; // Numero di punti da generare

    //unsigned long int seed = time(NULL);
    unsigned long int seed = 42;
    srand(seed);


    //punti CPU
    Point* points;
    points = (Point* )malloc(nPoints * sizeof(Point));
    readPoints("src/PRJ/points_CPU.csv", points);


    Point* centroids = (Point* )malloc(nCluster * sizeof(Point));
    generateRandomCentroids(centroids, nCluster);

    //Utilizzo gli stessi centroidi per entrambi i metodi
    Point* h_centroids = (Point* )malloc(nCluster * sizeof(Point));
    deepCopyPoints(h_centroids, centroids, nCluster);

    int labelsSEQ[nPoints];

    /***********************************************************/
	/*                     CPU clustering                      */
	/***********************************************************/
    start = seconds();
    kmeansCPU(points, nPoints, centroids, nCluster, labelsSEQ);
    end = seconds();
    double CPUtime = end - start;
    printf("Tempo di esecuzione CPU: %f\n",CPUtime);

    writePointsV2(points, labelsSEQ, nPoints, "src/PRJ/clusteredPoints_CPU.csv");
    free(points);
    /***********************************************************/
	/*                     GPU clustering                      */
	/***********************************************************/


    Point *h_points;
    h_points = (Point* )malloc(nPoints * sizeof(Point));
    readPoints("src/PRJ/points_CPU.csv", h_points);

    Point *d_points;
    cudaMalloc(&d_points, nPoints * sizeof(Point));

    cudaMemcpy(d_points, h_points, nPoints * sizeof(Point), cudaMemcpyHostToDevice);


    //generazione centroidi
    Point* d_centroids;
    cudaMalloc(&d_centroids, nCluster * sizeof(Point));

    //trasferimento centroidi da host a device
    cudaMemcpy(d_centroids, h_centroids, nCluster * sizeof(Point), cudaMemcpyHostToDevice);

    //generazione array label trasferimento non necessario
    int* h_labelsPARALLEL = (int* ) malloc(nPoints * sizeof(int));

    int* d_labelsPARALLEL;
    cudaMalloc(&d_labelsPARALLEL, nPoints * sizeof(int));
    cudaMemset(d_labelsPARALLEL, 0, nPoints * sizeof(int)); // Inizializza a zero

    //generazione array count
    int* d_count;
    cudaMalloc(&d_count, nCluster * sizeof(int));
    cudaMemset(d_count, 0, nCluster * sizeof(int)); // Inizializza a zero

    //generazione array sum_points
    Point* d_sum_points;
    cudaMalloc(&d_sum_points, nCluster * sizeof(Point));
    cudaMemset(d_sum_points, 0, nCluster * sizeof(Point)); // Inizializza a zero

    //creazione della variabile d_changed
    int* d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    int blockSize = 64;
    int numBlocks = (nPoints + blockSize - 1) / blockSize;
    start = seconds();
    //print first d_centroids
    printf("%f, %f\n", h_centroids[0].x, h_centroids[0].y);
    do {
        cudaMemset(d_changed, 0, sizeof(int)); // Resetta changed
        //Primo passo: sommare i punti assegnati
        kmeansSumPoints<<<numBlocks, blockSize>>>(d_points, d_centroids, d_labelsPARALLEL, d_count, d_sum_points, nPoints, nCluster, d_changed);
        cudaDeviceSynchronize();

        //Secondo passo: calcolare le nuove coordinate dei centroidi
        kmeansUpdateCentroids<<<1, nCluster>>>(d_centroids, d_sum_points, d_count, nCluster);
        cudaDeviceSynchronize();

        cudaMemset(d_sum_points, 0, nCluster * sizeof(Point)); // Resetta sum_points
        cudaMemset(d_count, 0, nCluster * sizeof(int)); // Resetta count
        cudaMemcpy(h_centroids, d_centroids, nCluster * sizeof(Point), cudaMemcpyDeviceToHost);

        int h_changed;
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_changed == 0) break;

    } while (true);
    printf("%f, %f\n", h_centroids[0].x, h_centroids[0].y);
    endNoTransf = seconds();
    // Copia dei risultati da device a host
    cudaMemcpy(h_points, d_points, nPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labelsPARALLEL, d_labelsPARALLEL, nPoints * sizeof(int), cudaMemcpyDeviceToHost);

    endTransf = seconds();
    double GPUtimeNoTransf = endNoTransf - start;
    double GPUtimeTransf = endTransf - start;
    printf("Tempo di esecuzione GPU (NO Transfer): %f\n",GPUtimeNoTransf);
	  double speedupNoTransf = CPUtime/GPUtimeNoTransf;
	  printf("    Speedup (NO transfer) %.1f\n", speedupNoTransf);

    printf("Tempo di esecuzione GPU (with Transfer): %f\n",GPUtimeTransf);
	  double speedupTransf = CPUtime/GPUtimeTransf;
	  printf("    Speedup (with transfer) %.1f\n", speedupTransf);



    // Scrittura dei risultati su file
    writePointsV2(h_points, h_labelsPARALLEL, nPoints, "src/PRJ/clusteredPoints_GPU.csv");

    // Cleanup
    cudaFree(d_points);

    cudaFree(d_centroids);
    cudaFree(d_labelsPARALLEL);
    cudaFree(d_count);
    cudaFree(d_sum_points);
    cudaFree(d_changed);
    free(h_labelsPARALLEL);

    return 0;
}
