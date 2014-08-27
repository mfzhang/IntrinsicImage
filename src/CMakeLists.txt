CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
FIND_PACKAGE( OpenCV REQUIRED )

INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(../include/superpixel)
SET(source ./test.cc ./main.cc )
ADD_EXECUTABLE(intrinsic.out ${source})
TARGET_LINK_LIBRARIES(intrinsic.out ${OpenCV_LIBS})
