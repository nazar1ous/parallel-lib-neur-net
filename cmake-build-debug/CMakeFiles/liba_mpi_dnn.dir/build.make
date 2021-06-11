# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2021.1.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion-2021.1.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dlutchyn/Documents/UCU/parallel-lib-neur-net

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/liba_mpi_dnn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/liba_mpi_dnn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/liba_mpi_dnn.dir/flags.make

CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.o: CMakeFiles/liba_mpi_dnn.dir/flags.make
CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.o: ../src/models/dnn_model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dlutchyn/Documents/UCU/parallel-lib-neur-net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.o -c /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/src/models/dnn_model.cpp

CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/src/models/dnn_model.cpp > CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.i

CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/src/models/dnn_model.cpp -o CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.s

# Object files for target liba_mpi_dnn
liba_mpi_dnn_OBJECTS = \
"CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.o"

# External object files for target liba_mpi_dnn
liba_mpi_dnn_EXTERNAL_OBJECTS =

libliba_mpi_dnn.a: CMakeFiles/liba_mpi_dnn.dir/src/models/dnn_model.cpp.o
libliba_mpi_dnn.a: CMakeFiles/liba_mpi_dnn.dir/build.make
libliba_mpi_dnn.a: CMakeFiles/liba_mpi_dnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dlutchyn/Documents/UCU/parallel-lib-neur-net/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libliba_mpi_dnn.a"
	$(CMAKE_COMMAND) -P CMakeFiles/liba_mpi_dnn.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/liba_mpi_dnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/liba_mpi_dnn.dir/build: libliba_mpi_dnn.a

.PHONY : CMakeFiles/liba_mpi_dnn.dir/build

CMakeFiles/liba_mpi_dnn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/liba_mpi_dnn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/liba_mpi_dnn.dir/clean

CMakeFiles/liba_mpi_dnn.dir/depend:
	cd /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dlutchyn/Documents/UCU/parallel-lib-neur-net /home/dlutchyn/Documents/UCU/parallel-lib-neur-net /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/cmake-build-debug /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/cmake-build-debug /home/dlutchyn/Documents/UCU/parallel-lib-neur-net/cmake-build-debug/CMakeFiles/liba_mpi_dnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/liba_mpi_dnn.dir/depend

