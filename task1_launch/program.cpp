#include "program.h"

#include "../task1/task1lib.h"

int main(int argc, char** argv) {
    if (GetDensityMatricesDiagonalsSequence(argc, argv)) {
        return 1;
    }

    return 0;
}
