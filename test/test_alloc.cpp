#include <gtest/gtest.h>
#include "alloc.h"

TEST(AllocTest, BasicTest1) {
  EXPECT_EQ(add(1, 2), 3);  // 示例断言
}

TEST(AllocTest, BasicTest2) {
  EXPECT_EQ(add(1, 1), 2);  // 示例断言
}
