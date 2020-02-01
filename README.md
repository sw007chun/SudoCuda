serial_solver.cpp: Sequential DFS
dfs_parallel.cu: Parallel DFS
dfs_probing.cu: DFS with parallel probing

Three source files compiles to separate programs on each on. 

To compile:
```
module load cuda-10.2
nvcc dfs_parallel.cu -o dfs_parallel -std=c++11
nvcc dfs_probing.cu -o dfs_probing -std=c++11
g++ serial_solver.cpp -o serial_solver

```

To use:
All three programs accepts Sudoku board from STDIN. The Sudoku board is N^4 integers separate by space or line break. 0 represents empty cells, and [1, N^2] represents given number clues.

Two cuda versions accept an optional program argument for N (default to 3). 
The sequential version doesn't accept N as program argument. Instead one must edit the macro in the beginning of the source file to change N.

Example:
```
echo "1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0" | ./dfs_probing 3
```
