set(TEST_EXE_NAME "A0Tests")

set(SHARED_TESTS_PATH "${CMAKE_CURRENT_LIST_DIR}/../shared_tests")

add_executable(${TEST_EXE_NAME}
	"main.cpp"
	"${SHARED_TESTS_PATH}/grading_helpers.h" 
	"cpp_tests.h"
	)
target_include_directories(${TEST_EXE_NAME} PRIVATE "../src/") # Get access to "your_code_here.h"
target_include_directories(${TEST_EXE_NAME} PRIVATE "${SHARED_TESTS_PATH}") # Get access to "grading_helpers.h"
target_compile_features(${TEST_EXE_NAME} PRIVATE cxx_std_20)
set_project_warnings(${TEST_EXE_NAME}) 
enable_sanitizers(${TEST_EXE_NAME}) 

# OpenMP support.
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TEST_EXE_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(${TEST_EXE_NAME} PRIVATE "-DDATA_DIR=\"${SHARED_TESTS_PATH}/data\"")

target_link_libraries(${TEST_EXE_NAME} PRIVATE CGFramework Catch2WithMain glm::glm)

# Copy all files in the shared test files folder to the build directory after every successful build.
add_custom_command(TARGET ${TEST_EXE_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${SHARED_TESTS_PATH}/data" $<TARGET_FILE_DIR:${TEST_EXE_NAME}>/data)
