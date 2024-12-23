#include "task2.h"

#include "mpi.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void fillVectorFromFile(std::vector<double>& w, const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Ошибка открытия файла: " << filename << std::endl;
        return;
    }
    for (auto& val : w) {
        if (!(ifs >> val)) break;
    }
}

unsigned long long factorial(int n) {
    unsigned long long res = 1;
    for (int i = 2; i <= n; ++i) {
        res *= i;
    }
    return res;
}

unsigned long long binomialCoefficient(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    return factorial(n) / (factorial(k) * factorial(n - k));
}

void CalcLocalRange(int rank, int nprocs, int N, int& rowsCount, int& offset) {
    int base = N / nprocs, extra = N % nprocs;
    rowsCount = base + (rank < extra);
    offset = (rank < extra)
             ? rank * (base + 1)
             : extra * (base + 1) + (rank - extra) * base;
}

double* GetLocalH(int rank, int nprocs, int ctx,
                const std::vector<double>& a,
                const std::vector<double>& w,
                int N, int Emax, int i,
                const std::vector<int>& sizes)
{
    int ones = Emax - i;
    int N2 = 1 << N;
    int localSize = sizes[i];

    std::vector<int> allStates(localSize);
    for (int j = 0, idx = 0; j < N2; ++j) {
        if (__builtin_popcount(j) == ones) {
            allStates[idx++] = j;
        }
    }

    int rowsCount, offset;
    CalcLocalRange(rank, nprocs, localSize, rowsCount, offset);

    double* H_i_loc = new double[localSize * rowsCount]();

    for (int j = offset; j < offset + rowsCount; j++) {
        for (int k = 0; k < localSize; k++) {
            if (j == k) {
                unsigned tmp = allStates[j];
                double val = 0;
                int bit_idx = 0;
                while (tmp) {
                    if (tmp & 1u) {
                        val += w[N - bit_idx - 1];
                    }
                    tmp >>= 1;
                    bit_idx++;
                }
                H_i_loc[(j - offset) * localSize + k] = val;
            } else {
                int x = allStates[j] ^ allStates[k];
                if (__builtin_popcount(x) != 2) {
                    continue;
                }
                double val = 0;
                int bit_idx = 0;
                while (x) {
                    if (x & 1) {
                        x >>= 1;
                        val = (x & 1) ? a[N - bit_idx - 2] : 0;
                    } else {
                        x >>= 1;
                    }
                    bit_idx++;
                }
                H_i_loc[(j - offset) * localSize + k] = val;
            }
        }
    }

    double* localH = rank == 0 ? new double[localSize * localSize] : nullptr;
    for (int p = 0; p < nprocs; ++p) {
        Cblacs_barrier(ctx, "All");
        if (rank == p) {
            Cdgesd2d(ctx, rowsCount, localSize, H_i_loc, rowsCount, 0, 0);
        }
        if (rank == 0) {
            int lrows, offs;
            CalcLocalRange(p, nprocs, localSize, lrows, offs);
            Cdgerv2d(ctx, lrows, localSize, localH + offs * localSize, lrows, 0, p);
        }
    }

    delete[] H_i_loc;
    return localH;
}

void Gather(bool is_root, int Emax, int Emin, int N,
                    double* H, const double* localH, int i, const std::vector<int>& sizes)
{
    if (!is_root) return;

    int start = 0;
    for (int j = 0; j < i; j++) {
        start += sizes[j];
    }

    int size_loc = sizes[i];
    int full_size = start;
    for (int j = i; Emax - j >= Emin; ++j) {
        full_size += sizes[j];
    }

    for (int row = 0; row < size_loc; ++row) {
        std::copy(localH + row * size_loc,
                  localH + (row + 1) * size_loc,
                  H + (start + row) * full_size + start);
    }
}

double* GetGlobalH(int rank, int nproc, int ctx,
                         std::vector<double>& a, std::vector<double>& w,
                         int N, int Emax, int Emin, std::vector<int>& sizes)
{
    int n = 0;
    for (int i = 0; Emax - i >= Emin; i++) {
        n += sizes[i];
    }

    double* H = (rank == 0 ? new double[n * n] : nullptr);

    for (int i = 0; Emax - i >= Emin; ++i) {
        double* localH = GetLocalH(rank, nproc, ctx, a, w, N, Emax, i, sizes);
        Gather(rank == 0, Emax, Emin, N, H, localH, i, sizes);
        delete[] localH;
    }
    return H;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int nprocs = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool mpiroot = (rank == 0);

    if (argc < 7) {
        if (mpiroot) {
            std::cerr << "Usage: ./mpirun -np <nprocs> N a_file w_file k Emin Emax\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::stringstream stream;
    int N, k, Emin, Emax;
    stream << argv[1] << " " << argv[4] << " " << argv[5] << " " << argv[6];
    stream >> N >> k >> Emin >> Emax;

    std::string fnameA(argv[2]);
    std::string fnameW(argv[3]);

    std::vector<double> a(N - 1), w(N);
    fillVectorFromFile(a, fnameA);
    fillVectorFromFile(w, fnameW);

    if (mpiroot) {
        std::cout << "A: ";
        for (auto val : a) std::cout << val << ' ';
        std::cout << "\nW: ";
        for (auto val : w) std::cout << val << ' ';
        std::cout << "\n";
    }

    int n = 0;
    std::vector<int> sizes(k + 1);
    for (int i = 0; Emin <= Emax - i; i++) {
        sizes[i] = binomialCoefficient(N, Emax - i);
        n += sizes[i];
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    int nprow = dims[0], npcol = dims[1], ctx = 0;
    Cblacs_get(-1, 0, &ctx);
    Cblacs_gridinit(&ctx, const_cast<char*>("R"), nprow, npcol);

    double* globalH = GetGlobalH(rank, nprocs, ctx, a, w, N, Emax, Emin, sizes);

    if (mpiroot) {
        std::vector<std::complex<double>> H(n * n);
        for (int i = 0; i < n * n; ++i) {
            H[i] = std::complex<double>(globalH[i], 0.0);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << H[i * n + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    Cblacs_gridexit(ctx);
    Cblacs_exit(0);
    return 0;
}
