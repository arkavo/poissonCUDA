#include <iostream>

int main() {
    #pragma acc parallel loop
    std::cout << "Hello World!";
    return 0;
}