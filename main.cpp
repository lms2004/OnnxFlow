#include "alloc.h"
#include <memory>


int main(){
    std::allocator<int> alloc;
    int* p = alloc.allocate(5);
}