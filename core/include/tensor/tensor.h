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
