CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(SubHarmonicTest)

FIND_PACKAGE(ArrayFire)
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(CUDA)

INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(SubHarmonicTest phase_screen_test.cu getSubHarmonic.cu subtract2DMean.cu makeComplexPSD.cu fftshift.cu getComplexReal.cu)
TARGET_LINK_LIBRARIES(SubHarmonicTest ${ArrayFire_CUDA_LIBRARIES} ${OpenCV_LIBS} dl curand)
