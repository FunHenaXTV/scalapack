#include "program.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include "mpi.h"

class TScopeLogger {
   public:
    TScopeLogger(const char* scopeName, int size)
        : StartTime(std::chrono::high_resolution_clock::now())
    {
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

template <typename T>
std::ostream& operator<<(std::ostream& stream, TMatrix<T>& matrix) {
    auto cols = matrix.GetCols();
    auto rows = matrix.GetRows();

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            stream << std::setw(3) << *(matrix.Data() + rows * c + r) << " ";
        }
        stream << std::endl;
    }

    std::flush(stream);

    return stream;
}

template <typename T>
TMatrix<T> operator*(TMatrix<T>& leftMatrix,
                     TMatrix<T>& rightMatrix) {
    if (leftMatrix.GetCols() != rightMatrix.GetRows()) {
        std::stringstream ss;
        ss << "Wrong matrix sizes " << leftMatrix.GetCols() << " "
           << leftMatrix.GetRows() << " " << rightMatrix.GetCols() << " "
           << leftMatrix.GetRows();

        throw ss.str();
    }

    TMatrix<T> result(leftMatrix.GetRows(), rightMatrix.GetCols());

    auto rows = leftMatrix.GetRows();
    auto cols = rightMatrix.GetCols();

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            *(result.Data() + c * rows + r) = 0;
            for (int k = 0; k < leftMatrix.GetCols(); k++) {
                auto a = *(leftMatrix.Data() + k * rows + r);
                auto b = *(rightMatrix.Data() + c * leftMatrix.GetCols() + k);
                *(result.Data() + c * rows + r) += a * b;
            }
        }
    }

    return result;
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
                Cdgesd2d(context, nr, nc, globalA.Data() + N * c + r, N, sendr,
                         sendc);
            }

            if (myrow == sendr && mycol == sendc) {
                Cdgerv2d(context, nr, nc, localA.Data() + nrows * recvc + recvr,
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
                Cdgesd2d(context, nr, nc, localA.Data() + nrows * recvc + recvr,
                         nrows, 0, 0);
                recvc = (recvc + nc) % ncols;
            }

            if (mpiroot) {
                Cdgerv2d(context, nr, nc, result.Data() + N * c + r, N, sendr,
                         sendc);
            }
        }

        if (myrow == sendr) {
            recvr = (recvr + nr) % nrows;
        }
    }

    return result;
}

int main(int argc, char** argv) {
    int zero = 0;
    int one = 1;
    int info = 0;

    int NA = 5, MA = 2, NB = 2, MB = 3;
    int NBlocksA = 2, MBlocksA = 1;
    int NBlocksB = 1, MBlocksB = 1;

    MPI_Init(&argc, &argv);
    int nprocs = 0, myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    bool mpiroot = (myrank == 0);

    TMatrix<double> globalA(NA, MA);
    TMatrix<double> globalB(NB, MB);
    TMatrix<double> checkRes(NA, MB);
    if (mpiroot) {
        {
            TScopeLogger scopeLogger("fill_matrix", sizeof("fill_matrix"));

            globalA.RandomFill(1);
            globalB.RandomFill(2);
            std::cout << "Matrix A:" << std::endl;
            std::cout << globalA << std::endl;

            std::cout << "Matrix B:" << std::endl;
            std::cout << globalB << std::endl;
        }

        {
            TScopeLogger scopeLogger("multiply_matricies",
                                     sizeof("multiply_matricies"));
            checkRes = globalA * globalB;
            std::cout << "CheckRes matrix" << std::endl;
            std::cout << checkRes << std::endl;
        }
    }

    int dims[2];
    MPI_Dims_create(nprocs, 2, dims);
    int nprow = dims[0];  // cartesian direction 0
    int npcol = dims[1];  // cartesian direction 1

    int context = 0;
    char order = 'R';

    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, &order, nprow, npcol);

    int myrow, mycol;
    int nprows, npcols;
    Cblacs_gridinfo(context, &nprows, &npcols, &myrow, &mycol);

    int NBlockA, MBlockA, NBlockB, MBlockB;
    int nrowsA, ncolsA, nrowsB, ncolsB;

    auto localA = ScatterMatrix(mpiroot, context, NA, MA, NBlocksA, MBlocksA,
                                NBlockA, MBlockA, nrowsA, ncolsA, globalA);

    auto localB = ScatterMatrix(mpiroot, context, NB, MB, NBlocksB, MBlocksB,
                                NBlockB, MBlockB, nrowsB, ncolsB, globalB);

    for (int i = 0; i < nprocs; i++) {
        std::flush(std::cout);
        if (i == myrank) {
            std::cout << "Proc: " << i << std::endl;
            std::cout << localA << std::endl;
            std::cout << localB << std::endl;
        }
        std::flush(std::cout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    auto nrowsC = numroc_(&NA, &NBlockA, &myrow, &zero, &nprows);
    auto ncolsC = numroc_(&MB, &MBlockB, &mycol, &zero, &npcols);

    // Create descriptors
    int rsrc = 0, csrc = 0;
    int* desca = new int[9];
    int* descb = new int[9];
    int* descc = new int[9];
    descinit_(desca, &NA, &MA, &NBlockA, &MBlockA, &rsrc, &csrc, &context,
              &nrowsA, &info);
    descinit_(descb, &NB, &MB, &NBlockB, &MBlockB, &rsrc, &csrc, &context,
              &nrowsB, &info);
    descinit_(descc, &NA, &MB, &NBlockA, &MBlockB, &rsrc, &csrc, &context,
              &nrowsC, &info);

    double alpha = 1.0;
    double betta = 0;

    // Distributed multiplication
    TMatrix<double> localC(nrowsC, ncolsC);

    char notrans = 'N';
    pdgemm_(&notrans, &notrans, &NA, &MB, &MA, &alpha, localA.Data(), &one,
            &one, desca, localB.Data(), &one, &one, descb, &betta,
            localC.Data(), &one, &one, descc);

    int NBlocksC = NBlocksA, MBlocksC = MBlocksB;
    auto globalC =
        GatherMatrix(mpiroot, context, NA, MB, NBlocksC, MBlocksC, localC);

    if (mpiroot) {
        std::cout << globalC << std::endl;
    }

    delete[] desca;
    delete[] descb;
    delete[] descc;

    Cblacs_gridexit(context);
    Cblacs_exit(0);

    return 0;
}