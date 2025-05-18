#include <memory>
#include <string>
#include <vector>
#include "base.h"
#include "buffer.h"

/*
* Tensor::Tensor: 构造函数
* parameters:
*   - data_type: 数据类型
*   - dim0, dim1, dim2, dim3/dims_: 张量的维度
*   - need_alloc: 是否需要分配内存
*   - alloc: 内存分配器
*   - ptr: 外部内存指针
*/
class Tensor{
public:
    // ----------- 构造函数 -----------
    explicit Tensor() = default;

    explicit Tensor(DataType data_type, int32_t dim0, bool need_alloc = false,
                    std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                    std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    bool need_alloc = false, std::shared_ptr<DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                    bool need_alloc = false, std::shared_ptr<DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    explicit Tensor(DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                    std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);


    // ----------- core func -----------
    void reset(DataType data_type, const std::vector<int32_t>& dims);
    void reshape(const std::vector<int32_t>& dims);
    void resize(const std::vector<int32_t>& dims);

    bool allocate(std::shared_ptr<DeviceAllocator> alloc = nullptr, bool need_realloc = false);
    void init_buffer(std::shared_ptr<DeviceAllocator> alloc, DataType data_type,
                         bool need_alloc, void* ptr);
    bool assign(std::shared_ptr<Buffer> buffer);

    void to_cuda(cudaStream_t stream = nullptr);
    void to_cpu();
    
    // ----------- template func -----------
    template <typename T>
    T* ptr();

    template <typename T>
    const T* ptr() const;

    template <typename T>
    T* ptr(int64_t index);

    template <typename T>
    const T* ptr(int64_t index) const;

    template <typename T>
    T& index(int64_t offset);

    template <typename T>
    const T& index(int64_t offset) const;

    // -------- getter func --------
    size_t size() const;
    int32_t dims_size() const;

    DeviceType device_type() const;
    DataType data_type() const;

    const std::vector<int32_t>& dims() const;
    std::shared_ptr<Buffer> get_buffer() const;

    // -------- attribute func --------
    bool is_empty() const;
    size_t byte_size() const;

    int32_t get_dim(int32_t idx) const;
    std::vector<size_t> strides() const;
    
    // --------- setter func ---------
    void set_device_type(DeviceType device_type) const;


private:
    size_t size_ = 0;  // 张量中数据的个数。例如，如果张量存储了三个元素 {1, 2, 3}，则 size_ = 3。
    
    std::vector<int32_t> dims_;  // 张量的维度。例如，如果张量是二维的且维度为 {2, 3}，则 dims_ 存储 {2, 3}，表示张量有 2 行 3 列。

    std::shared_ptr<Buffer> buffer_;  // 用于存储张量数据的内存缓冲区。
    
    DataType data_type_ = DataType::kDataTypeUnknown; 
};

template <typename T>
T& Tensor::index(int64_t offset) {
  Assert(offset < this->size() && offset >= 0, "Index out of bounds");
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  Assert(offset < this->size() && offset >= 0, "Index out of bounds");
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
    Assert(buffer_ != nullptr && buffer_->ptr() != nullptr, 
           "The data area buffer of this tensor is empty or it points to a null pointer.");
    return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
    Assert(buffer_ != nullptr && buffer_->ptr() != nullptr, 
           "The data area buffer of this tensor is empty or it points to a null pointer.");
    return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}