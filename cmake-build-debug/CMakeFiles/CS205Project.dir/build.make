# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\CLion\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\CLion\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/CS205Project.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CS205Project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CS205Project.dir/flags.make

CMakeFiles/CS205Project.dir/main.cpp.obj: CMakeFiles/CS205Project.dir/flags.make
CMakeFiles/CS205Project.dir/main.cpp.obj: CMakeFiles/CS205Project.dir/includes_CXX.rsp
CMakeFiles/CS205Project.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CS205Project.dir/main.cpp.obj"
	D:\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\CS205Project.dir\main.cpp.obj -c "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\main.cpp"

CMakeFiles/CS205Project.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CS205Project.dir/main.cpp.i"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\main.cpp" > CMakeFiles\CS205Project.dir\main.cpp.i

CMakeFiles/CS205Project.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CS205Project.dir/main.cpp.s"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\main.cpp" -o CMakeFiles\CS205Project.dir\main.cpp.s

# Object files for target CS205Project
CS205Project_OBJECTS = \
"CMakeFiles/CS205Project.dir/main.cpp.obj"

# External object files for target CS205Project
CS205Project_EXTERNAL_OBJECTS =

CS205Project.exe: CMakeFiles/CS205Project.dir/main.cpp.obj
CS205Project.exe: CMakeFiles/CS205Project.dir/build.make
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_gapi452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_highgui452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_ml452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_objdetect452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_photo452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_stitching452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_video452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_videoio452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_dnn452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_imgcodecs452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_calib3d452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_features2d452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_flann452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_imgproc452.dll.a
CS205Project.exe: D:/opencv/opencv/mingw-build/install/x64/mingw/lib/libopencv_core452.dll.a
CS205Project.exe: CMakeFiles/CS205Project.dir/linklibs.rsp
CS205Project.exe: CMakeFiles/CS205Project.dir/objects1.rsp
CS205Project.exe: CMakeFiles/CS205Project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CS205Project.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\CS205Project.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CS205Project.dir/build: CS205Project.exe

.PHONY : CMakeFiles/CS205Project.dir/build

CMakeFiles/CS205Project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\CS205Project.dir\cmake_clean.cmake
.PHONY : CMakeFiles/CS205Project.dir/clean

CMakeFiles/CS205Project.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project" "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project" "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\cmake-build-debug" "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\cmake-build-debug" "D:\Uni File\2021Spring\C&Cpp Programing\CS205Project\CS205Project_6.10\CS205Project\cmake-build-debug\CMakeFiles\CS205Project.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/CS205Project.dir/depend

