
#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define at(arr, x, y) (arr[(x) * size + (y)])
#define maximum_t 32

double epsilon = 1E-6;
int maximum_iteration = 1E6;
int size = 128;
bool show_matrix = false;
int update_matrix = 100;
//выполняет один шаг вычислений для двумерной матрицы A на
// графическом процессоре (GPU), и записывает результат в двумерную матрицу B.
//Функция является ядром (kernel) для выполнения на GPU, что
// означает, что она будет выполняться параллельно на множестве
// нитей (threads), каждая из которых будет обрабатывать свой сегмент
// данных.
//В этой функции, индексы нити находятся с помощью блока индексов
// (blockIdx) и индексов нитей в блоке (threadIdx), и используются
// для нахождения индексов соответствующих элементов в матрицах A и B.
//Далее, проверяется, не находится ли текущий элемент на границе
// матрицы, и если это так, функция завершается без выполнения вычислений.
//Если же текущий элемент не находится на границе, то производится
// вычисление нового значения элемента в матрице B с использованием
// значений соседних элементов из матрицы A.
//Новое значение элемента находится как среднее арифметическое
// четырех соседних элементов в матрице A, расположенных слева, справа,
// сверху и снизу от текущего элемента.
__global__ void computeOneStepOnGPU(double *A, double *B, int size) {
    // эти строки кода переводят координаты нити внутри блока и индекс блока
    // в индексы элементов в матрице A и B, над которыми будет производиться операция.
    int i = blockIdx.y * blockDim.y + threadIdx.y; //blockIdx имеет три поля: x, y и z,
    // которые содержат индексы блока нитей по каждой оси.
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // чтобы не изменять границы
    if (j == 0 || i == 0 || i >= size-1 || j >= size-1)
        return;

    at(B, i, j) = 0.25 * (at(A, i-1, j) + at(A, i, j-1) + at(A, i+1, j) + at(A, i, j+1));
}
//провести параллельную операцию вычитания на каждом элементе массивов A и B.
// Блоки и нити используются для распараллеливания операции, что ускоряет ее выполнение.
__global__ void gpuMatrixSubtraction(double *A, double *B, int size) {
    //определяют индексы элементов A и B, которые будут обрабатываться текущим
    // потоком на основе значений blockIdx.y, blockDim.y, threadIdx.y, blockIdx.x,
    // blockDim.x и threadIdx.x
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if( i >= (size-1))
        return;
    if (j >= (size-1))
        return;
    at(B, i, j) = at(A, i, j) - at(B, i, j);
}

__global__ void gpuInitializeMatrixA(double *A, int size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//проверка if(i >= size-1) обеспечивает то, что потоки не выходят
// за границы матрицы. Если поток выходит за пределы границ матрицы,
// то выполнять дальнейшие вычисления нет необходимости и поток завершает свою работу
    if(i >= size-1)
        return;

    int step;
    step = 10.0 / (size - 1) * i;

    at(A, 0, i) = 10.0 + step;
    at(A, i, 0) = 10.0 + step;
    at(A, i, size-1) = 20.0 + step;
    at(A, size-1, i) = 20.0 + step;
}

int main(int argc, char **argv) {
    cudaSetDevice(3);
    for(int arg = 0; arg < argc; arg += 1) {
        if(0 == strcmp(argv[arg], "-error")) {
            epsilon = atof(argv[arg + 1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-iter")) {
            maximum_iteration = atoi(argv[arg + 1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-size")) {
            size = atoi(argv[arg+1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-update_matrix")) {
            update_matrix = atoi(argv[arg + 1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-show_matrix")) {
            show_matrix = true;
            arg += 1;
        }
    }

    int size_matrix = size * size;

    cudaError_t crush;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t     graph;
    cudaGraphExec_t g_instance;

    double *A_GPU;
    double *cudaError;
    double *tmp = NULL;
    double *A_GPU_NEW;

    // количество блоков в гриде и нитей в блоке соответственно
    dim3 blocks_in_grid   = dim3(size / maximum_t, size / maximum_t);
    dim3 threads_in_block = dim3(maximum_t, maximum_t);

    size_t temp_size = 0;

    cudaMalloc(&cudaError, sizeof(double) * 1);
    cudaMalloc(&A_GPU, sizeof(double) * size_matrix);
    cudaMalloc(&A_GPU_NEW, sizeof(double) * size_matrix);

    dim3 thread = size < 1024 ? size : 1024;
    dim3 block = size / (size < 1024 ? size : 1024);

    // заполнение матриц начальными значениями (значения границ)
    gpuInitializeMatrixA<<<block, thread>>>(A_GPU, size);
    gpuInitializeMatrixA<<<block, thread>>>(A_GPU_NEW, size);

    // вычисление tmp_size (размера tmp)
    //выполняет редукцию на устройстве CUDA, находя максимальное значение элементов
    // в переданном массиве A_GPU_NEW размера size_matrix. Результат сохраняется в переменной tmp.
    cub::DeviceReduce::Max(tmp, temp_size, A_GPU_NEW, cudaError, size_matrix, stream);
    //tmp - указатель на выделенный на устройстве CUDA буфер для хранения результата редукции;
    //temp_size - указатель на переменную, в которой должен быть передан размер буфера tmp. При вызове функции переменная temp_size должна содержать размер буфера в байтах, а после вызова функции будет содержать реальный размер буфера, который требуется для хранения результата редукции;
    //A_GPU_NEW - указатель на выделенный на устройстве CUDA буфер, содержащий данные для редукции;
    //cudaError - указатель на выделенную на устройстве CUDA переменную для хранения ошибок;
    //size_matrix - размер массива A_GPU_NEW;
    //stream - указатель на поток выполнения на устройстве CUDA, в котором будет выполняться редукция.
    cudaMalloc(&tmp, temp_size);


    // ================================================================================
    //запускает захват операций, которые будут выполняться в потоке stream.
    // Когда захват начинается, все операции, которые выполняются в потоке,
    // записываются в граф потока CUDA, вместо того чтобы быть выполненными
    // непосредственно на GPU. Это позволяет создать граф потока, который может
    //быть выполнен позже, синхронно или асинхронно.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);//Аргумент cudaStreamCaptureModeGlobal указывает, что будут записываться все операции, выполняемые в потоке stream, включая операции, которые были запущены до вызова cudaStreamBeginCapture.
    // выполнение шагов алгоритма столько раз, сколько необходимо до обновления ошибки
    for(int i = 0; i < update_matrix; i += 2)
    {
        computeOneStepOnGPU<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_GPU, A_GPU_NEW, size); // 0 - количество разделяемой памяти в байтах, которую ядро будет выделять на каждый блок потоков.
        // В данном случае значение 0 указывает на то, что ядро не требует выделения дополнительной разделяемой памяти.
        computeOneStepOnGPU<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_GPU_NEW, A_GPU, size);
    }

    gpuMatrixSubtraction<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_GPU, A_GPU_NEW, size);

    // вычисление ошибки
    cub::DeviceReduce::Max(tmp, temp_size, A_GPU_NEW, cudaError, size_matrix, stream);

    gpuInitializeMatrixA<<<block, thread, 0, stream>>>(A_GPU_NEW, size);
//завершает захват текущего потока воспроизведения CUDA и возвращает объект cudaGraph_t, представляющий граф потока.

    cudaStreamEndCapture(stream, &graph);
    //создает экземпляр графа CUDA с помощью cudaGraph_t,
    // полученного из cudaStreamEndCapture. Этот экземпляр графа
    // может быть запущен на устройстве, которое было использовано при захвате графа.

    cudaGraphInstantiate(&g_instance, graph, NULL, NULL, 0);

    // =================================================================================

    int iter = 0;
    double error = 1.0;

    while(error > epsilon && iter < maximum_iteration)
    {
        cudaGraphLaunch(g_instance, stream);
        cudaMemcpy(&error, cudaError, sizeof(double), cudaMemcpyDeviceToHost);
        iter += update_matrix;
    }

    std::cout << "iterations: " << iter  << std::endl;
    std::cout << "error: " << error << std::endl;

    // вывод матрицы
    if(show_matrix)
    {
        double* A = new double[size_matrix];
        cudaMemcpyAsync(&A, A_GPU, sizeof(double), cudaMemcpyDeviceToHost, stream);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++)
                std::cout << at(A_GPU, i, j) << ' ';
            std::cout << std::endl;
        }
    }

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaFree(A_GPU);
    cudaFree(A_GPU_NEW);
    cudaFree(tmp);
    return 0;
}

