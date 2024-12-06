#include "program.h"

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

void printDiagonals(std::vector<std::vector<double>>& diagonals) {
    std::cout << "Diagonals: " << std::endl;
    for (const auto& diag : diagonals) {
        for (const auto& value : diag) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
double* get_eigen_e_v(int N, int NBlocks, int MBlocks, TMatrix<T>& localH,
                      TMatrix<T>& localZ, const int* desch, const int* descz) {
    double* w = new double[N];

    char v = 'V';
    char u = 'U';
    std::vector<std::complex<double>> work(1), rwork(1);
    int lwork = -1, rlwork = -1;
    int info = 0;
    int one = 1;

    pzheev_(&v, &u, &N, localH.Data(), &one, &one, desch, w, localZ.Data(),
            &one, &one, descz, work.data(), &lwork,
            reinterpret_cast<double*>(rwork.data()), &rlwork, &info);
    if (info != 0) {
        std::cerr << "Error in pzheev, info = " << info << std::endl;
        exit(1);
    }

    lwork = work[0].real();
    rlwork = rwork[0].real();

    work.resize(size_t(lwork));
    rwork.resize(size_t(rlwork));

    pzheev_(&v, &u, &N, localH.Data(), &one, &one, desch, w, localZ.Data(),
            &one, &one, descz, work.data(), &lwork,
            reinterpret_cast<double*>(rwork.data()), &rlwork, &info);

    if (info != 0) {
        std::cerr << "Error in pzheev, info = " << info << std::endl;
        exit(1);
    }

    return w;
}

template <typename T>
void calc_diag(int n, int N, double dT, int nrowsH, int ncolsH, int MBlocksH,
               int NBlocksH, int MBlocksRo, int myrow, int mycol, int nprows,
               int npcols, int rsrc, int csrc, int context, TMatrix<T>& localZ,
               int* descz, bool mpiroot, TMatrix<T>& localRo, int* descro,
               int ncolsRo, std::vector<std::vector<double>>& diagonals,
               double* w, int nprocs, int myrank) {
    int zero = 0;
    int one = 1;
    int info = 0;

    std::complex<double> alpha = 1.0;
    std::complex<double> betta = 0;

    double t = 0;

    for (int step = 0; step <= n; ++step) {
        TMatrix<std::complex<double>> localDiagonalMatrix(nrowsH, ncolsH,
                                                          false);
        for (int iloc = 1; iloc <= nrowsH; iloc++) {
            for (int jloc = 1; jloc <= ncolsH; jloc++) {
                int i_glob = indxl2g_(&iloc, &MBlocksH, &myrow, &zero, &nprows);
                int j_glob = indxl2g_(&jloc, &NBlocksH, &mycol, &zero, &npcols);
                if (i_glob == j_glob) {
                    // Data_[N * c + r]
                    localDiagonalMatrix.Data()[(jloc - 1) * nrowsH + iloc - 1] =
                        exp(std::complex<double>(0, -w[i_glob - 1] * t / hbar));
                } else {
                    localDiagonalMatrix.Data()[(jloc - 1) * nrowsH + iloc - 1] =
                        0;
                }
            }
        }

        TMatrix<std::complex<double>> localC(nrowsH, ncolsH);

        // trash-code
        int* descd = new int[9];
        int* descc = new int[9];
        descinit_(descd, &N, &N, &NBlocksH, &MBlocksH, &rsrc, &csrc, &context,
                  &nrowsH, &info);
        descinit_(descc, &N, &N, &NBlocksH, &MBlocksH, &rsrc, &csrc, &context,
                  &nrowsH, &info);

        char notrans = 'N';
        char trans = 'C';
        pzgemm_(&notrans, &notrans, &N, &N, &N, (double*)&alpha, localZ.Data(),
                &one, &one, descz, localDiagonalMatrix.Data(), &one, &one,
                descd, (double*)&betta, localC.Data(), &one, &one, descc);

        TMatrix<std::complex<double>> localC2(nrowsH, ncolsH);
        int* descc2 = new int[9];
        descinit_(descc2, &N, &N, &NBlocksH, &MBlocksH, &rsrc, &csrc, &context,
                  &nrowsH, &info);

        pzgemm_(&notrans, &trans, &N, &N, &N, (double*)&alpha, localC.Data(),
                &one, &one, descc, localZ.Data(), &one, &one, descz,
                (double*)&betta, localC2.Data(), &one, &one, descc2);

        TMatrix<std::complex<double>> localC3(nrowsH, ncolsRo);
        int* descc3 = new int[9];
        descinit_(descc3, &N, &N, &NBlocksH, &MBlocksRo, &rsrc, &csrc, &context,
                  &nrowsH, &info);
        pzgemm_(&notrans, &notrans, &N, &N, &N, (double*)&alpha, localC2.Data(),
                &one, &one, descc2, localRo.Data(), &one, &one, descro,
                (double*)&betta, localC3.Data(), &one, &one, descc3);

        TMatrix<std::complex<double>> localC4(nrowsH, ncolsH);
        int* descc4 = new int[9];
        descinit_(descc4, &N, &N, &NBlocksH, &MBlocksH, &rsrc, &csrc, &context,
                  &nrowsH, &info);
        pzgemm_(&notrans, &trans, &N, &N, &N, (double*)&alpha, localC3.Data(),
                &one, &one, descc3, localC2.Data(), &one, &one, descc2,
                (double*)&betta, localC4.Data(), &one, &one, descc4);

        auto globalC4 =
            GatherMatrix(mpiroot, context, N, N, NBlocksH, MBlocksH, localC4);

        if (mpiroot) {
            std::vector<double> diag(N);
            for (int i = 0; i < N; ++i) {
                diag[i] = std::abs(globalC4.Data()[i * N + i]);
            }
            diagonals.push_back(diag);
        }

        delete[] descc;
        delete[] descd;
        delete[] descc2;
        delete[] descc3;
        delete[] descc4;

        t += dT;
    }
}

int initialization(int argc, char** argv) {
    int info = 0;

    int N, n;
    double dT;
    int NBlocksH = 1, MBlocksH = 1;
    int NBlocksRo = 1, MBlocksRo = 1;

    MPI_Init(&argc, &argv);
    int nprocs = 0, myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    bool mpiroot = (myrank == 0);

    if (argc < 6) {
        if (mpiroot) {
            std::cerr << "Usage: ./mpirun -np <nprocs> N matrix_file dT H.bin n"
                      << std::endl;
        }

        MPI_Finalize();
        return 1;
    }

    std::stringstream stream;
    stream << argv[1] << " " << argv[3] << " " << argv[5];
    stream >> N >> dT >> n;

    TMatrix<std::complex<double>> globalRo(N, N, false);
    TMatrix<std::complex<double>> globalH(N, N, false);

    std::string fname_ro(argv[2]);
    std::string fname_H(argv[4]);

    if (mpiroot) {
        TScopeLogger scopeLogger("fill_matrix", sizeof("fill_matrix"));
        globalRo.FillFromFile(fname_ro);

        std::cout << "Matrix ro:" << std::endl;
        std::cout << globalRo << std::endl;

        globalH.FillFromBinFile(fname_H);

        std::cout << "Matrix H:" << std::endl;
        std::cout << globalH << std::endl;
    }

    int dims[2];
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

    auto localRo =
        ScatterMatrix(mpiroot, context, N, N, NBlocksRo, MBlocksRo, NBlockRo,
                      MBlockRo, nrowsRo, ncolsRo, globalRo);

    auto localH = ScatterMatrix(mpiroot, context, N, N, NBlocksH, MBlocksH,
                                NBlockH, MBlockH, nrowsH, ncolsH, globalH);
    int rsrc = 0, csrc = 0;
    int* desch = new int[9];
    int* descz = new int[9];
    int* descro = new int[9];
    descinit_(desch, &N, &N, &NBlocksH, &MBlocksH, &rsrc, &csrc, &context,
              &nrowsH, &info);
    descinit_(descz, &N, &N, &NBlocksH, &MBlocksH, &rsrc, &csrc, &context,
              &nrowsH, &info);
    descinit_(descro, &N, &N, &NBlocksRo, &MBlocksRo, &rsrc, &csrc, &context,
              &nrowsRo, &info);

    TMatrix<std::complex<double>> localZ(nrowsH, ncolsH, false);
    double* w =
        get_eigen_e_v(N, NBlocksH, MBlocksH, localH, localZ, desch, descz);

    if (mpiroot) {
        std::cout << std::endl << "Eigenevalues: ";
        for (int i = 0; i < N; i++) {
            std::cout << w[i] << ' ';
        }
        std::cout << std::endl << std::endl;
    }

    // for (int i = 0; i < nprocs; i++) {
    //     std::flush(std::cout);
    //     if (i == myrank) {
    //         std::cout << "LocalZ Proc: " << i << std::endl;
    //         std::cout << localZ << std::endl;
    //     }
    //     std::flush(std::cout);
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // auto globalZ =
    //     GatherMatrix(mpiroot, context, N, N, NBlocksH, MBlocksH, localZ);
    // if (mpiroot) {
    //     std::cout << globalZ << std::endl;
    // }

    std::vector<std::vector<double>> diagonals;

    calc_diag(n, N, dT, nrowsH, ncolsH, MBlocksH, NBlocksH, MBlocksRo, myrow,
              mycol, nprows, npcols, rsrc, csrc, context, localZ, descz,
              mpiroot, localRo, descro, ncolsRo, diagonals, w, nprocs, myrank);

    if (mpiroot) {
        printDiagonals(diagonals);
    }

    delete[] desch;
    delete[] descro;
    delete[] descz;

    delete[] w;

    Cblacs_gridexit(context);
    Cblacs_exit(0);

    return 0;
}
int main(int argc, char** argv) {
    if (initialization(argc, argv)) {
        return 1;
    }

    return 0;
}
