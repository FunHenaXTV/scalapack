#include <fstream>
#include <iostream>
#include <complex>
#include <ctime>
#include <cstdlib>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <binary_output_file> <text_output_file>" << endl;
        return 1;
    }

    fstream binaryFile(argv[1], ios::out | ios::binary);
    fstream textFile(argv[2], ios::in);

    if (!binaryFile.is_open() || !textFile.is_open()) {
        cerr << "Error opening files." << endl;
        return 1;
    }

    int m = 4;
    int n = 4;

    long ltime = time(NULL);
    int stime = (unsigned int)ltime / 2;
    srand(stime);


    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // complex<double> complexNum(rand() % 201 - 100, rand() % 201 - 100); // Generate random complex number
            complex<double> complexNum;
            textFile >> complexNum;

            // cout << complexNum << endl;

            binaryFile.write(reinterpret_cast<char*>(&complexNum), sizeof(complexNum));  //Write complex number to binary file

            // textFile << complexNum << " "; //Write to text file.  << operator handles complex number output
        }
    }

    binaryFile.close();
    textFile.close();

    return 0;
}

