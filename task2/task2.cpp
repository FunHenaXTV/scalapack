#include "task2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "mpi.h"

template <typename T>
class TMatrix {
   public:
    TMatrix(const int n, const int m, const bool randomFill = true)
        : N(n), M(m), Data_(new T[N * M]) {
        if (randomFill) {
            RandomFill();
        }
    }

    TMatrix(TMatrix&& matrix) {
        Data_ = matrix.Data_;
        N = matrix.N;
        M = matrix.M;
    }

    TMatrix<T>& operator=(TMatrix<T>&& matrix) {
        if (this == &matrix) {
            return *this;
        }

        delete[] Data_;
        Data_ = std::exchange(matrix.Data_, nullptr);
        N = std::exchange(matrix.N, 0);
        M = std::exchange(matrix.M, 0);
        return *this;
    }

    void RandomFill(int alpha = 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-10.0, 10.0);

        int i = 0;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < M; c++) {
                Data_[N * c + r] = (i + 1) + (alpha * i);
                i++;
            }
        }
    }

    bool FillFromFile(const std::string& filename) {
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        for (int r = 0; r < N; r++) {
            for (int c = 0; c < M; c++) {
                file >> Data_[N * c + r];
            }
        }

        file.close();
        return true;
    }

    bool FillFromBinFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        // в файле хранится по строкам, записываем по столбцам
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < M; c++) {
                file.read(reinterpret_cast<char*>(Data_ + N * c + r),
                          sizeof(T));
            }
        }
        return true;
    }

    T* operator[](int pos) const {
        if (pos >= N) {
            throw "Out of bound";
        }
        return &Data_[pos * M];
    }

    T* Data() { return Data_; }

    int GetRows() const { return N; }

    int GetCols() const { return M; }

    ~TMatrix() { delete[] Data_; }

   private:
    int N, M;
    T* Data_;
};

const double hbar = 1.0;

class TScopeLogger {
   public:
    TScopeLogger(const char* scopeName, int size)
        : StartTime(std::chrono::high_resolution_clock::now()) {
        for (auto i = 0; i < size; i++) {
            ScopeName.push_back(scopeName[i]);
        }
    }

    ~TScopeLogger() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            endTime - StartTime);
        std::string scopeName(ScopeName.data(), ScopeName.size());
        std::cout << "Exiting scope: " << scopeName
                  << " (duration: " << duration.count() << " microseconds)"
                  << std::endl;
    }

   private:
    std::chrono::high_resolution_clock::time_point StartTime;
    std::vector<char> ScopeName;
};

void FillFromVector(const TMatrix<std::complex<double>>& vector,
                    TMatrix<std::complex<double>>& result) {
    for (int i = 0; i < result.GetCols(); i++) {
        for (int j = 0; j < result.GetCols(); j++) {
            result[i][j] = vector[i][0] * std::conj(vector[j][0]);
        }
    }
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, TMatrix<T>& matrix) {
    int rows = matrix.GetRows();
    int cols = matrix.GetCols();
    int width = 10;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            stream << std::setw(25) << *(matrix.Data() + rows * c + r) << " ";
        }
        stream << std::endl;
    }

    std::flush(stream);

    return stream;
}

template <typename T>
TMatrix<T> ScatterMatrix(bool mpiroot, int context, int N, int M, int NBlocks,
                         int MBlocks, int& NBlock, int& MBlock, int& nrows,
                         int& ncols, TMatrix<T>& globalA) {
    int zero = 0;

    int myrow, mycol;
    int nprows, npcols;
    Cblacs_gridinfo(context, &nprows, &npcols, &myrow, &mycol);

    NBlock = N / nprows;
    MBlock = M / npcols;

    nrows = numroc_(&N, &NBlock, &myrow, &zero, &nprows);
    ncols = numroc_(&M, &MBlock, &mycol, &zero, &npcols);

    int sendr = 0, sendc = 0;
    int recvr = 0, recvc = 0;

    TMatrix<T> localA(nrows, ncols);

    for (int r = 0; r < N; r += NBlocks, sendr = (sendr + 1) % nprows) {
        sendc = 0;

        int nr = NBlocks;
        if (N - r < NBlocks) {
            nr = N - r;
        }

        for (int c = 0; c < M; c += MBlocks, sendc = (sendc + 1) % npcols) {
            int nc = MBlocks;
            if (M - c < MBlocks) {
                nc = M - c;
            }

            if (mpiroot) {
                Czgesd2d(context, nr, nc, globalA.Data() + N * c + r, N, sendr,
                         sendc);
            }

            if (myrow == sendr && mycol == sendc) {
                Czgerv2d(context, nr, nc, localA.Data() + nrows * recvc + recvr,
                         nrows, 0, 0);
                recvc = (recvc + nc) % ncols;
            }
        }

        if (myrow == sendr) {
            recvr = (recvr + nr) % nrows;
        }
    }

    return localA;
}

template <typename T>
TMatrix<T> GatherMatrix(bool mpiroot, int context, int N, int M, int NBlocks,
                        int MBlocks, TMatrix<T>& localA) {
    int zero = 0;

    int myrow, mycol;
    int nprows, npcols;
    Cblacs_gridinfo(context, &nprows, &npcols, &myrow, &mycol);

    int NBlock = N / nprows;
    int MBlock = M / npcols;

    int nrows = numroc_(&N, &NBlock, &myrow, &zero, &nprows);
    int ncols = numroc_(&M, &MBlock, &mycol, &zero, &npcols);

    int sendr = 0, sendc = 0;
    int recvr = 0, recvc = 0;

    TMatrix<T> result(N, M);
    for (int r = 0; r < N; r += NBlocks, sendr = (sendr + 1) % nprows) {
        sendc = 0;
        int nr = NBlocks;
        if (N - r < NBlocks) {
            nr = N - r;
        }

        for (int c = 0; c < M; c += MBlocks, sendc = (sendc + 1) % npcols) {
            int nc = MBlocks;
            if (M - c < MBlocks) {
                nc = M - c;
            }

            if (myrow == sendr && mycol == sendc) {
                Czgesd2d(context, nr, nc, localA.Data() + nrows * recvc + recvr,
                         nrows, 0, 0);
                recvc = (recvc + nc) % ncols;
            }

            if (mpiroot) {
                Czgerv2d(context, nr, nc, result.Data() + N * c + r, N, sendr,
                         sendc);
            }
        }

        if (myrow == sendr) {
            recvr = (recvr + nr) % nrows;
        }
    }

    return result;
}

void fillVectorFromFile(std::vector<double>& w, const std::string& filename) {
    std::ifstream inputFile(filename);

    if (!inputFile) {
        std::cerr << "Ошибка открытия файла: " << filename << std::endl;
        return;
    }

    double number;
    size_t index = 0;

    while (index < w.size() && inputFile >> number) {
        w[index] = number;
        index++;
    }

    inputFile.close();
}

unsigned long long factorial(int n) {
    unsigned long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

unsigned long long binomialCoefficient(int n, int k) {
    if (k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    return factorial(n) / (factorial(k) * factorial(n - k));
}

// double* gen_H_i(context& ctxt, int n_i, int i){
//     double* H_i_loc;
//     int* vecs;

//     Partition part = compute_partition(ctxt.mpirank, ctxt.nprocs, n_i);

//     H_i_loc = new double[n_i * part.local_rows];
//     int n_ = 0;
//     vecs = new int[n_i];
//     for (int j = 0; j < N2; j++) {
//         if (__builtin_popcount(j) == ones) {
//             vecs[n_++] = j;
//             // cout << bitset<32>(j).to_string().substr(32-p.N, p.N) << " ";
//         }
//     }

//     for (int j = part.offset; j < part.offset + part.local_rows; j++) {
//         for (int k = 0; k < n_i; k++) {
//             if (j == k) {
//                 double x = 0;
//                 int index = 0;
//                 while (vecs[j] > 0) {
//                     if (vecs[j] & 1) {
//                         x += p.w[p.N - index-1];
//                     }
//                     vecs[j] >>= 1;
//                     index++;
//                 }
//                 H_i_loc[(j-part.offset)*n_i + k] = x;
//             } else {
//                 if (__builtin_popcount(vecs[j]^vecs[k]) != 2)
//                     H_i_loc[(j-part.offset)*n_i + k] = 0;

//                 int number = vecs[j]^vecs[k];
//                 int index = 0;
//                 while (number > 0) {
//                     if (number & 1) {
//                         number >>= 1;
//                         H_i_loc[(j-part.offset)*n_i + k] = (number & 1) ?
//                         p.a[p.N-index-2] : 0;
//                     }
//                     number >>= 1;
//                     index++;
//                 }
//                 H_i_loc[(j-part.offset)*n_i + k] = 0;
//             }
//         }
//     }

//     double* H_i = NULL;
//     if (ctxt.is_root) {
//         H_i = new double[n_i*n_i];
//         for (int j = 0; j < n_i; j++)
//             for (int k = 0; k< n_i; k++)
//                 H_i[j*n_i+k] = 0;
//     }
//     for (int proc = 0; proc < ctxt.nprocs; proc++){
//         Cblacs_barrier(ctxt.ctxt, "All");
//         if (ctxt.mpirank == proc) {
//             Cdgesd2d(ctxt.ctxt, part.local_rows, n_i, H_i_loc,
//             part.local_rows, 0, 0);
//         }
//         if (ctxt.is_root) {
//             Partition loc_part = compute_partition(proc, ctxt.nprocs, n_i);
//             Cdgerv2d(ctxt.ctxt, loc_part.local_rows, n_i,
//             H_i+n_i*loc_part.offset, loc_part.local_rows, 0, proc);
//         }
//     }

//     delete[] H_i_loc;
//     delete[] vecs;
//     return H_i;
// }

// void write_H_i_in_H(context& ctxt, Parameters& p, double* H, double* H_i, int
// i){
//     if (!ctxt.is_root)
//         return;

//     int start = 0;
//     int size_loc = get_size_i(p, i);
//     for (int j = 0; j < i; j++) {
//         start += get_size_i(p, j);
//     }
//     int size = start;
//     for (int j = i; p.Emax - j >= p.Emin; j++) {
//         size += get_size_i(p, j);
//     }

//     for (int j = 0; j < size_loc; j++) {
//         for (int k = 0; k < size_loc; k++) {
//             H[(j+start)*size + start + k] = H_i[j*size_loc + k];
//         }
//     }
// }

int main(int argc, char** argv) {
    int info = 0;

    int N, k, Emin, Emax;
    int NBlocks = 2, MBlocks = 2;

    MPI_Init(&argc, &argv);
    int nprocs = 0, myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    bool mpiroot = (myrank == 0);

    if (argc < 7) {
        if (mpiroot) {
            std::cerr
                << "Usage: ./mpirun -np <nprocs> N a_file w_file k Emin Emax"
                << std::endl;
        }

        MPI_Finalize();
        return 1;
    }

    std::stringstream stream;
    stream << argv[1] << " " << argv[4] << " " << argv[5] << " " << argv[6];
    stream >> N >> k >> Emin >> Emax;

    std::string fnameA(argv[2]);
    std::string fnameW(argv[3]);

    std::vector<double> a(N - 1);
    std::vector<double> w(N);

    fillVectorFromFile(a, fnameA);
    fillVectorFromFile(w, fnameW);

    if (mpiroot) {
        std::cout << "A: ";
        for (int i = 0; i < N - 1; i++) {
            std::cout << a[i] << ' ';
        }
        std::cout << "\nW: ";
        for (int i = 0; i < N; i++) {
            std::cout << w[i] << ' ';
        }
        std::cout << "\n";
    }

    int n = 0;
    std::vector<int> n_i(k + 1);
    for (int i = 0; Emin <= Emax - i; i++) {
        n_i[i] = binomialCoefficient(N, Emax - i);
        n += n_i[i];
    }
    TMatrix<std::complex<double>> globalH(n, n, false);

    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    int nprow = dims[0];
    int npcol = dims[1];
    int context = 0;
    char order = 'R';

    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, &order, nprow, npcol);

    int myrow, mycol;
    int nprows, npcols;
    Cblacs_gridinfo(context, &nprows, &npcols, &myrow, &mycol);

    int NBlockRo, MBlockRo, NBlockH, MBlockH;
    int nrowsRo, ncolsRo, nrowsH, ncolsH;

    // for (int i = 0; Emin <= Emax - i; i++) {
    //     double* H_i = gen_H_i(ctxt, p, i);

    //     write_H_i_in_H(ctxt, p, H, H_i, i);

    //     delete[] H_i;
    // }

    Cblacs_gridexit(context);
    Cblacs_exit(0);

    return 0;
}