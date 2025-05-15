#include "myfunc.h"

void printError(const char* errorMessage) {
    if (errorMessage == nullptr) {
        std::cout << "\033[31mError: Invalid error message\033[0m" << std::endl;
        return;
    }
    // ANSI 转义码：\033[31m 设置文本为红色，\033[0m 重置文本颜色为默认
    std::cout << "\033[31mError: " << errorMessage << "\033[0m" << std::endl;
}

void* Malloc(size_t size){
    void* ptr = NULL;

    if(size == 0){
        printError("malloc size is 0");
        return NULL;
    }

    ptr = malloc(size);
    
    if(ptr == NULL){
        printError("malloc failed");

    }
    return ptr;
}
