// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "unitaryspace_testfixture.h"

#include "gtest/gtest.h"

#if defined(__AVX512F__) && !defined(_WIN32)

#include "../lib/formux.h"
#include "../lib/unitaryspace_avx512.h"

namespace qsim {
namespace unitary {
namespace {

TEST(UnitarySpaceAVX512Test, SetZero) {
  TestSetZeros<UnitarySpaceAVX512<For>>();
}

TEST(UnitarySpaceAVX512Test, SetIdentity) {
  TestSetIdentity<UnitarySpaceAVX512<For>>();
}

TEST(UnitarySpaceAVX512Test, GetEntry) {
  TestSetEntry<UnitarySpaceAVX512<For>>();
}

}  // namspace
}  // namespace unitary
}  // namespace qsim

#endif  // __AVX512F__ && !_WIN32

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
