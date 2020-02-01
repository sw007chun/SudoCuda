
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <algorithm>

typedef unsigned char u8;

constexpr unsigned CONFLICT = 0xFFFFFFFF;
constexpr unsigned SOLVED = 0xFFFFFFFE;

template<typename T, unsigned rank>
__device__
unsigned Propagate(T board[], unsigned pivots[], unsigned pivots_count,
    T output[][rank * rank * rank * rank], unsigned output_pivot[], unsigned* output_count) {
    constexpr unsigned width = rank * rank;
    while(pivots_count != 0) {
        unsigned pivot = pivots[--pivots_count];
        unsigned row = pivot / width;
        unsigned col = pivot % width;
        unsigned crow = row / rank;
        unsigned ccol = col / rank;
        unsigned irow = row % rank;
        unsigned icol = col % rank;
        unsigned mask = ~board[pivot];

#define Do(r, c) { \
    unsigned position = (r) * width + (c); \
    T old = board[position]; \
    unsigned left = __popcll(board[position] = old & mask); \
    if (left == 0) return CONFLICT; \
    if (left == 1 && left < __popcll(old)) { \
        pivots[pivots_count++] = position; \
    } \
}

        for (unsigned i = 0; i < width - 1; ++i) {
            Do(i >= row ? i + 1 : i, col);
            Do(row, i >= col ? i + 1 : i);
        }

        for (unsigned frow = 0; frow < rank - 1; ++frow) {
            for (unsigned fcol = 0; fcol < rank - 1; ++fcol) {
                Do(crow * rank + (frow >= irow ? frow + 1 : frow),
                   ccol * rank + (fcol >= icol ? fcol + 1 : fcol));
            }
        }
    }

    unsigned pivot = SOLVED;
    unsigned value = 0xFFFFFFFF;
    for (unsigned i = 0; i < width * width; ++i) {
        unsigned new_value = __popcll(board[i]);
        if (new_value > 1 && new_value < value) {
            value = new_value;
            pivot = i;
            break;
        }
    }

    if (pivot == SOLVED) {
        return SOLVED;
    }

    for (unsigned i = 0; i < value; ++i) {
        unsigned output_i = atomicAdd(output_count, 1);
        output_pivot[output_i] = pivot;
        for (unsigned pos = 0; pos < width * width; ++pos) {
            if (pos == pivot) {
                T bit = (T)1 << (__ffsll(board[pos]) - 1);
                output[output_i][pos] = bit;
                board[pos] &= ~bit;
            } else {
                output[output_i][pos] = board[pos];
            }
        }
    }
    return 0;
}

template<typename T, unsigned rank>
__global__
void Search(unsigned count, T stack[][rank * rank * rank * rank], unsigned pivot_stack[], unsigned* stack_size,
    T board_buffer[][rank * rank * rank * rank], unsigned pivot_transfer[], unsigned* finish_signal) {

    unsigned id = blockIdx.x * gridDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned pivot_buffer[100]{pivot_transfer[id]};

    if (Propagate<T, rank>(board_buffer[id], pivot_buffer, 1, stack, pivot_stack, stack_size) ==SOLVED) {
        *finish_signal = id;
    }
}

template<typename T, unsigned rank>
__global__
void Init(int* clue, T stack[][rank * rank * rank * rank], unsigned pivot_stack[], unsigned* stack_size,
    T board_buffer[][rank * rank * rank * rank], unsigned* finish_signal) {
    constexpr unsigned width = rank * rank;
    unsigned pivot_buffer[width * width]{};
    unsigned pivot_count = 0;

    *stack_size = 0;

    for (unsigned i = 0; i < width * width; ++i) {
        if (clue[i] == 0) {
            board_buffer[0][i] = (T)(((T)1 << width) - 1);
        } else {
            board_buffer[0][i] = (T)1 << (clue[i] - 1);
            pivot_buffer[pivot_count++] = i;
        }
    }

    if (Propagate<T, rank>(board_buffer[0], pivot_buffer, pivot_count, stack, pivot_stack, stack_size) ==SOLVED) {
        *finish_signal = 0;
    } else {
        *finish_signal = 0xFFFFFFFF;
    }
}

#define checkError(code) { checkErrorImpl((code), __FILE__, __LINE__); }
void checkErrorImpl(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::printf("[%s: %d]CUDA Error: %s\n", file, line, cudaGetErrorString(code));
        exit(-1);
    }
}

template<typename T, unsigned rank>
void SolveImpl(int* clue) {
    constexpr unsigned width = rank * rank;
    constexpr unsigned count = width * width;
    int *clue_gpu;
    checkError(cudaMalloc(&clue_gpu, count * sizeof(int)));
    checkError(cudaMemcpy(clue_gpu, clue, count * sizeof(int), cudaMemcpyHostToDevice));

    T (*stack)[count];
    checkError(cudaMalloc(&stack, 1024 * 1024 * 1024));
    unsigned *pivot_stack;
    checkError(cudaMalloc(&pivot_stack, 2 * 1024 * 1024));
    unsigned *stack_size;
    checkError(cudaMalloc(&stack_size, sizeof(unsigned)));

    constexpr unsigned thread_count = 1024;
    unsigned *pivot_transfer;
    checkError(cudaMalloc(&pivot_transfer, sizeof(unsigned) * thread_count));
    T (*board_buffer)[count];
    checkError(cudaMalloc(&board_buffer, sizeof(T) * count * thread_count));
    unsigned *finish_signal;
    checkError(cudaMalloc(&finish_signal, sizeof(unsigned)));

    Init<T, rank><<<1, 1>>>(clue_gpu, stack, pivot_stack, stack_size,
        board_buffer, finish_signal);

    unsigned finish_signal_cpu;
    checkError(cudaMemcpy(&finish_signal_cpu, finish_signal, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (finish_signal_cpu != 0xFFFFFFFFu) goto finish;

    unsigned stack_size_cpu;
    while(true) {
        checkError(cudaMemcpy(&stack_size_cpu, stack_size, sizeof(unsigned), cudaMemcpyDeviceToHost));
        // printf("%u\n", stack_size_cpu);
        if (stack_size_cpu == 0) {
            printf("Conflict\n");
            return;
        }
        unsigned batch = std::min(stack_size_cpu, thread_count);
        checkError(cudaMemcpy(board_buffer, stack + stack_size_cpu - batch, batch * sizeof(T) * count, cudaMemcpyDeviceToDevice));
        checkError(cudaMemcpy(pivot_transfer, pivot_stack + stack_size_cpu - batch, batch * sizeof(unsigned), cudaMemcpyDeviceToDevice));
        stack_size_cpu -= batch;
        checkError(cudaMemcpy(stack_size, &stack_size_cpu, sizeof(unsigned), cudaMemcpyHostToDevice));

        Search<T, rank><<<1, batch>>>(batch, stack, pivot_stack, stack_size,
            board_buffer, pivot_transfer, finish_signal);

        checkError(cudaMemcpy(&finish_signal_cpu, finish_signal, sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (finish_signal_cpu != 0xFFFFFFFFu) goto finish;
    }

    finish:
    T board[count];
    checkError(cudaMemcpy(board, board_buffer[finish_signal_cpu], sizeof(T) * count, cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; ++i) {
        if (__builtin_popcountll(board[i]) == 1) {
            clue[i] = __builtin_ffsll(board[i]);
        } else {
            clue[i] = 0;
        }
    }
}


void Solve(unsigned rank, int* clue) {
    switch(rank) {
    case 2:
        SolveImpl<unsigned char, 2>(clue);
        break;
    case 3:
        SolveImpl<unsigned short, 3>(clue);
        break;
    case 4:
        SolveImpl<unsigned short, 4>(clue);
        break;
    case 5:
        SolveImpl<unsigned int, 5>(clue);
        break;
    case 6:
        SolveImpl<unsigned long long, 6>(clue);
        break;
    }
}

int main(int argc, char** argv) {
    unsigned rank = 3;
    if (argc >= 2) {
        rank = std::atoi(argv[1]);
    }

    unsigned width = rank * rank;

    int* clue = new int[width * width];

    for (unsigned i = 0; i < width * width; ++i) {
        int input;
        if (1 != std::scanf("%d", &input)) {
            input = 0;
        }
        clue[i] = input;
    }

    Solve(rank, clue);

    for (int row = 0; row < width; ++row) {
        if (row % rank == 0) {
            std::printf("\n");
        }
        for (int col = 0; col < width; ++col) {
            if (col % rank == 0) {
                std::printf("  ");
            }
            std::printf("%2d ", clue[row * width + col]);
        }
        std::printf("\n");
    }
}
