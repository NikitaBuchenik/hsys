add_executable(matrix_tests src/matrix_tests.cpp)
target_link_libraries(matrix_tests PRIVATE matrix_core eigen_headers GTest::gtest_main)
add_test(NAME matrix_tests COMMAND matrix_tests)
