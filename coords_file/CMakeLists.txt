
add_subdirectory(xdrfile)

add_library(coords_file coords_file.cpp)
target_link_libraries(coords_file xdrfile)

