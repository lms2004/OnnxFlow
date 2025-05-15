#include <memory>


template<class T>
class allocator {
public:
    using value_type = T; // 分配对象类型
    using size_type = size_t; // 内存大小类型
    using difference_type = ptrdiff_t; // 指针差值类型

    virtual allocator() = default; // 默认构造函数

    // 分配内存
    virtual T* allocate(size_t n) = 0;
    // 释放内存
    virtual void deallocate(T* p, size_t n) = 0;
};

