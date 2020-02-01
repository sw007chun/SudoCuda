
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <algorithm>

typedef unsigned char u8;

constexpr unsigned CONFLICT = 0xFFFFFFFF;
constexpr unsigned SOLVED = 0xFFFFFFFE;
constexpr unsigned INVALID_POS = 0xFFFFFFFD;

template <typename T, unsigned rank, unsigned pivot_buffer_max>
__device__
void Propagate(T (&board)[rank * rank * rank * rank],
    unsigned (&pivots)[pivot_buffer_max], unsigned* pivot_count, unsigned pivot, T mask) {

    constexpr unsigned width = rank * rank;
    // Calculate neighbor positions
    unsigned row = pivot / width;
    unsigned col = pivot % width;
    unsigned square_row = row / rank;
    unsigned square_col = col / rank;

    unsigned index = threadIdx.x % (width - 1);
    if (threadIdx.x < width - 1) {
        // first (width - 1) threads take care of cells in the same row
        col = index + (index >= col);
    } else if (threadIdx.x < 2 * width - 2) {
        // next (width - 1) threads take care of cells in the same column
        row = index + (index >= row);
    } else if (threadIdx.x < 2 * width - 2 + (rank - 1) * (rank - 1)) {
        // next (rank - 1)^2 threads take care of cells in the same square
        unsigned new_row = index / (rank - 1) + square_row * rank;
        unsigned new_col = index % (rank - 1) + square_col * rank;
        row = new_row + (new_row >= row);
        col = new_col + (new_col >= col);
    } else {
        return;
    }
    // Remove candidates from neighbor and counts the rest candidate
    unsigned neighbor = row * width + col;
    unsigned old = __popcll(board[neighbor]);
    unsigned left = __popcll(board[neighbor] &= mask);

    if (left < 2 && left < old) {
        unsigned i = atomicAdd(pivot_count, 1);
        if (i >= pivot_buffer_max) {
            printf("Pivot buffer overflow");
            return;
        }
        // left = 0 -> conflict
        // left = 1 -> new pivot
        pivots[i] = CONFLICT * (1 - left) + neighbor * left;
    }
}

template <typename T, unsigned rank, unsigned pivot_buffer_max>
__device__
bool Reduce(T (&board)[rank * rank * rank * rank],
    unsigned (&pivots)[pivot_buffer_max], unsigned* pivot_count) {
    while(*pivot_count != 0) {
        // Fetch the next pivot
        __syncthreads();
        unsigned pivot = pivots[*pivot_count - 1];
        __syncthreads();
        if (threadIdx.x == 0) {
            --*pivot_count;
        }
        __syncthreads();

        // If previously encountered a conflict, abort
        if (pivot == CONFLICT) {
            if (threadIdx.x == 0) {
                *pivot_count = 0;
            }
            return false;
        }

        __syncthreads();

        Propagate<T, rank, pivot_buffer_max>(board, pivots, pivot_count, pivot, ~board[pivot]);
        __syncthreads();
    }
    return true;
}

template <unsigned result_length>
__device__ __host__
void MinMerge(u8 (&value_a)[result_length], unsigned (&pos_a)[result_length],
    u8 (&value_b)[result_length], unsigned (&pos_b)[result_length]) {
    // Insertion sort to avoid extra space
    unsigned a_index = 0;
    for (unsigned b_index = 0; b_index < result_length; ++b_index) {
        while(true) {
            if (a_index == result_length) {
                return;
            }
            if (value_b[b_index] > value_a[a_index]) {
                ++a_index;
            } else {
                break;
            }
        }

        for (unsigned a_i2 = result_length - 1; a_i2 > a_index; --a_i2) {
            pos_a[a_i2] = pos_a[a_i2 - 1];
            value_a[a_i2] = value_a[a_i2 - 1];
        }

        pos_a[a_index] = pos_b[b_index];
        value_a[a_index] = value_b[b_index];
        ++a_index;
    }

}

// Finds result_length minimal candidates on the board,
// ignoring decided cells
template <typename T, unsigned rank, unsigned result_length>
__device__
void FindMins(T (&board)[rank * rank * rank * rank],
    u8 (&values)[rank * rank * rank * rank][result_length],
    unsigned (&pos)[rank * rank * rank * rank][result_length]) {
    constexpr unsigned count = rank * rank * rank * rank;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        u8 count = (u8)__popcll(board[i]);
        if (count == 1) {
            values[i][0] = 0xFF;
            pos[i][0] = INVALID_POS;
        } else {
            values[i][0] = count;
            pos[i][0] = i;
        }
        for (unsigned j = 1; j < result_length; ++j) {
            values[i][j] = 0xFF;
            pos[i][j] = INVALID_POS;
        }
    }

    __syncthreads();
    unsigned step = 1 << (31 - __clz(count - 1));

    for (; step > 0; step >>= 1) {
        unsigned limit = min(step, count - step);
        for (unsigned i = threadIdx.x; i < limit; i += blockDim.x) {
            MinMerge<result_length>(values[i], pos[i], values[i + step], pos[i + step]);
        }
        __syncthreads();
    }
}

template <typename T, unsigned rank, unsigned result_length>
__global__
void FindMinsKernel(T (&board)[rank * rank * rank * rank], unsigned (&result_pos)[result_length]) {

    __shared__ u8 values[rank * rank * rank * rank][result_length];
    __shared__ unsigned pos[rank * rank * rank * rank][result_length];

    FindMins<T, rank, result_length>(board, values, pos);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < result_length; i += blockDim.x) {
        result_pos[i] = pos[0][i];
    }
}

template <typename T, unsigned rank, unsigned max_depth, unsigned pivot_buffer_max>
__global__
void SearchTree(T (&board)[rank * rank * rank * rank],
    T output_boards_start[][rank * rank * rank * rank], unsigned* total, unsigned first_candidate) {
    constexpr unsigned width = rank * rank;
    auto output_boards = output_boards_start;

    // DFS stack. Each stack frame consists of
    // board_stack[*]: snapshot of the board in reduced state
    // candidate_stack[*]: position of the cell being branched on
    // bit_stack[*]: bit flags of the branching cell.
    //        Candidates that have been tried or being tried is removed from the flags
    __shared__ T board_stack[max_depth][width * width];
    __shared__ unsigned candidate_stack[max_depth - 1];
    __shared__ T bit_stack[max_depth - 1];

    unsigned stack_size = 1; // [1, max_depth)

    // scratch spaces for Reducer
    __shared__ unsigned pivots[pivot_buffer_max];
    __shared__ unsigned pivot_count;

    // scratch spaces for popcnt min
    __shared__ u8 min_value_board[width * width][1];
    __shared__ unsigned min_pos_board[width * width][1];

    // Bring in the initial board
    for (unsigned i = threadIdx.x; i < width * width; i += blockDim.x) {
        board_stack[0][i] = board[i];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        pivot_count = 0;
        *total = 0;
        candidate_stack[0] = first_candidate;
        bit_stack[0] = board_stack[0][first_candidate];
    }

    while(true) {
        // At this point, the stack top [stack_size - 1] contains
        // a reduced board, the cell to branch on, and the candidates
        // that hasn't been tried.

        __syncthreads();
        if (bit_stack[stack_size - 1] == 0) { // no more candidates
            if (stack_size == 1) { // stack is going to be empty
                break;
            }
            stack_size--;
            continue;
        }

        // Copy the board to a new stack frame
        for (unsigned i = threadIdx.x; i < width * width; i += blockDim.x) {
            board_stack[stack_size][i] = board_stack[stack_size - 1][i];
        }

        // Choose the branch
        T mask = ~((T)1 << (__ffsll(bit_stack[stack_size - 1]) - 1));
        __syncthreads();
        if (threadIdx.x == 0) {
            bit_stack[stack_size - 1] &= mask; // remove from candidate set
            board_stack[stack_size][candidate_stack[stack_size - 1]] = ~mask; // Fix the candidate cell
        }
        __syncthreads();

        // Propagate from the chosen branch
        Propagate<T, rank, pivot_buffer_max>(board_stack[stack_size], pivots, &pivot_count, candidate_stack[stack_size - 1], mask);
        __syncthreads();
        // Reduce the rest
        if (!Reduce<T, rank, pivot_buffer_max>(board_stack[stack_size], pivots, &pivot_count)) {
            // if the reduce found a conflict, we discard this branch and the new stack frame
            continue;
        }
        __syncthreads();
        if (stack_size == max_depth - 1) {
            // Reached the stack top, output
            for (unsigned i = threadIdx.x; i < width * width; i += blockDim.x) {
                (*output_boards)[i] = board_stack[stack_size][i];
            }
            output_boards += 1;
        } else {
            // Prepare for deeper stack.
            FindMins<T, rank, 1>(board_stack[stack_size], min_value_board, min_pos_board);
            __syncthreads();

            if (min_pos_board[0][0] == INVALID_POS) {
                // We actually found a solution. Dump to output and abort
                for (unsigned i = threadIdx.x; i < width * width; i += blockDim.x) {
                    (*output_boards_start)[i] = board_stack[stack_size][i];
                }
                if (threadIdx.x == 0) {
                    *total = SOLVED;
                }
                return;
            } else {
                // make the new stack frame a real frame
                if (threadIdx.x == 0) {
                    candidate_stack[stack_size] = min_pos_board[0][0];
                    bit_stack[stack_size] = board_stack[stack_size][candidate_stack[stack_size]];
                }
                stack_size++;
            }
        }
    }


    if (threadIdx.x == 0) {
        *total = output_boards - output_boards_start;
    }
}

template <typename T, unsigned rank, unsigned pivot_buffer_max>
__global__
void Initialize(T (&output_board)[rank * rank * rank * rank],
    int (&clue)[rank * rank * rank * rank], bool *init_success) {
    constexpr unsigned width = rank * rank;
    __shared__ T board[width * width];
    __shared__ unsigned pivots[pivot_buffer_max];
    __shared__ unsigned pivot_count;

    if (threadIdx.x == 0) {
        pivot_count = 0;
        *init_success = true;
    }

    for (unsigned i = threadIdx.x; i < width * width; i += blockDim.x) {
        board[i] = (T)(((T)1 << width) - 1);
    }

    __syncthreads();

    for (unsigned i = 0; i < width * width; i += 1) {
        if (clue[i] == 0) continue;
        T flags = (T)1 << (clue[i] - 1);
        if (threadIdx.x == 0) {
            board[i] = flags;
        }
        __syncthreads();
        Propagate<T, rank, pivot_buffer_max>(board, pivots, &pivot_count, i, ~flags);
        __syncthreads();
        if (!Reduce<T, rank, pivot_buffer_max>(board, pivots, &pivot_count)) {
            if (threadIdx.x == 0) {
                *init_success = false;
            }
            return;
        }
        __syncthreads();
    }

    for (unsigned i = threadIdx.x; i < width * width; i += blockDim.x) {
        output_board[i] = board[i];
    }
}

template <typename T, unsigned rank>
__global__
void Finalize(T (&board)[rank * rank * rank * rank],
    int (&clue)[rank * rank * rank * rank]) {
    constexpr unsigned width = rank * rank;
    for (unsigned i = threadIdx.x; i < width * width; i += blockDim.x) {
        if (__popcll(board[i]) == 1) {
            clue[i] = __ffsll(board[i]);
        } else {
            clue[i] = 0;
        }
    }
}

#define checkError(code) { checkErrorImpl((code), __FILE__, __LINE__); }
void checkErrorImpl(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::printf("[%s: %d]CUDA Error: %s\n", file, line, cudaGetErrorString(code));
        exit(-1);
    }
}

template <typename T, unsigned rank, unsigned pivot_buffer_max, unsigned max_depth, unsigned max_guess>
void SolveImpl(int *clue) {
    constexpr unsigned width = rank * rank;

    // Transfer clue table to gpu
    int (*clue_gpu)[width * width];
    checkError(cudaMalloc((void**)&clue_gpu, sizeof(int) * width * width));
    checkError(cudaMemcpy(clue_gpu, clue, sizeof(int) * width * width, cudaMemcpyHostToDevice));

    // Stack of boards
    T (*master_stack)[width * width];
    checkError(cudaMalloc((void**)&master_stack, 1024 * 1024 * 1024));

    // arrays of buffers. Each array for a guess point
    T (*(guess_buffer[max_guess]))[width * width];
    for (unsigned i = 0; i < max_guess; ++i) {
        checkError(cudaMalloc((void**)&guess_buffer[i], 512 * 1024 * 1024));
    }

    // array of counters for receiving the solution count
    unsigned (*total_counter_buffer)[max_guess];
    checkError(cudaMalloc((void**)&total_counter_buffer, max_guess * sizeof(unsigned)));

    // Buffer for scanning guess points
    unsigned (*min_pos_board)[max_guess];
    checkError(cudaMalloc((void**)&min_pos_board, max_guess * sizeof(unsigned)));
    unsigned min_pos_board_cpu[max_guess];

    // Streams for running each guesses
    cudaStream_t stream[max_guess];
    for (unsigned i = 0; i < max_guess; ++i) {
        checkError(cudaStreamCreate(&stream[i]));
    }

    // For receiving the init result
    bool init_success, *init_success_gpu;
    checkError(cudaMalloc((void**)&init_success_gpu, sizeof(bool)));

    // Initialize at stack bottom
    Initialize<T, rank, pivot_buffer_max><<<1, std::min(width * width, 1024u)>>>
        (master_stack[0], *clue_gpu, init_success_gpu);
    checkError(cudaMemcpy(&init_success, init_success_gpu, sizeof(bool), cudaMemcpyDeviceToHost));
    if (!init_success) {
        std::printf("Found conflict in init!\n");
    }

    // Boards available in the stack. We have an initial one already
    unsigned master_stack_size = 1;

    if (init_success) while(true) {
        //printf("%u\n", master_stack_size);
        // scan guess points for the stack top
        FindMinsKernel<T, rank, max_guess><<<1, std::min(width * width, 1024u)>>>
            (master_stack[master_stack_size - 1], *min_pos_board);

        checkError(cudaMemcpy(min_pos_board_cpu, min_pos_board,
            max_guess * sizeof(unsigned), cudaMemcpyDeviceToHost));

        if (min_pos_board_cpu[0] == INVALID_POS) {
            // No guess point. The board is actually solved
            break;
        }

        // dispatch solver for each guess point
        unsigned guess = 0;
        for (; guess < max_guess && min_pos_board_cpu[guess] != INVALID_POS; ++guess) {
            SearchTree<T, rank, max_depth, pivot_buffer_max>
                <<<1, std::min(width * width, 256u), 0, stream[guess]>>>
                (master_stack[master_stack_size - 1], guess_buffer[guess],
                &(*total_counter_buffer)[guess], min_pos_board_cpu[guess]);
        }

        checkError(cudaDeviceSynchronize());

        // find the guess that has fewest result and pump to the stack
        unsigned total_counter_buffer_cpu[max_guess];
        checkError(cudaMemcpy(total_counter_buffer_cpu,
            total_counter_buffer, guess * sizeof(unsigned),
            cudaMemcpyDeviceToHost));

        // check if ther is already a solved board
        auto solved = std::find(
            total_counter_buffer_cpu,
            total_counter_buffer_cpu + guess, SOLVED)
             - total_counter_buffer_cpu;
        if (solved != guess) {
            checkError(cudaMemcpy(
                &master_stack[master_stack_size - 1],
                guess_buffer[solved],
                width * width * sizeof(T), cudaMemcpyDeviceToDevice));
            break;
        }

        unsigned min_guess = std::min_element(
            total_counter_buffer_cpu,
            total_counter_buffer_cpu + guess)
             - total_counter_buffer_cpu;

        checkError(cudaMemcpy(
            &master_stack[master_stack_size - 1],
            guess_buffer[min_guess],
            width * width * total_counter_buffer_cpu[min_guess] * sizeof(T),
            cudaMemcpyDeviceToDevice));
        master_stack_size += total_counter_buffer_cpu[min_guess];
        master_stack_size -= 1;
        if (master_stack_size == 0) {
            std::printf("No solution!\n");
            master_stack_size = 1;
            break;
        }
    }

    Finalize<T, rank><<<1, std::min(width * width, 1024u)>>>
        (master_stack[master_stack_size - 1], *clue_gpu);

    checkError(cudaMemcpy(
        clue, clue_gpu,
        width * width * sizeof(int), cudaMemcpyDeviceToHost
    ));

    checkError(cudaFree(init_success_gpu));
    checkError(cudaFree(min_pos_board));
    checkError(cudaFree(total_counter_buffer));
    checkError(cudaFree(master_stack));
    checkError(cudaFree(clue_gpu));
    for (unsigned i = 0; i < max_guess; ++i) {
        checkError(cudaFree(guess_buffer[i]));
        checkError(cudaStreamDestroy(stream[i]));
    }
}

void Solve(unsigned rank, int* clue) {
    switch(rank) {
    case 2:
        SolveImpl<unsigned char, 2, 100, 4, 4>(clue);
        break;
    case 3:
        SolveImpl<unsigned short, 3, 100, 4, 4>(clue);
        break;
    case 4:
        SolveImpl<unsigned short, 4, 100, 4, 4>(clue);
        break;
    case 5:
        SolveImpl<unsigned int, 5, 100, 4, 4>(clue);
        break;
    case 6:
        SolveImpl<unsigned long long, 6, 100, 4, 4>(clue);
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
