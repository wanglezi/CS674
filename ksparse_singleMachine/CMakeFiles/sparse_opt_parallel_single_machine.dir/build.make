# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/lezi/tools/clion-2016.2.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/lezi/tools/clion-2016.2.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lezi/ClionProjects/ksparse_singleMachine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release

# Include any dependencies generated for this target.
include CMakeFiles/sparse_opt_parallel_single_machine.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sparse_opt_parallel_single_machine.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sparse_opt_parallel_single_machine.dir/flags.make

CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o: CMakeFiles/sparse_opt_parallel_single_machine.dir/flags.make
CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o: /home/lezi/ClionProjects/ksparse_singleMachine/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o -c /home/lezi/ClionProjects/ksparse_singleMachine/main.cpp

CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lezi/ClionProjects/ksparse_singleMachine/main.cpp > CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.i

CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lezi/ClionProjects/ksparse_singleMachine/main.cpp -o CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.s

CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.requires

CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.provides: CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/sparse_opt_parallel_single_machine.dir/build.make CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.provides

CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.provides.build: CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o


CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o: CMakeFiles/sparse_opt_parallel_single_machine.dir/flags.make
CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o: /home/lezi/ClionProjects/ksparse_singleMachine/src/common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o -c /home/lezi/ClionProjects/ksparse_singleMachine/src/common.cpp

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lezi/ClionProjects/ksparse_singleMachine/src/common.cpp > CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.i

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lezi/ClionProjects/ksparse_singleMachine/src/common.cpp -o CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.s

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.requires:

.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.requires

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.provides: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.requires
	$(MAKE) -f CMakeFiles/sparse_opt_parallel_single_machine.dir/build.make CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.provides.build
.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.provides

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.provides.build: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o


CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o: CMakeFiles/sparse_opt_parallel_single_machine.dir/flags.make
CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o: /home/lezi/ClionProjects/ksparse_singleMachine/src/solve.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o -c /home/lezi/ClionProjects/ksparse_singleMachine/src/solve.cpp

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lezi/ClionProjects/ksparse_singleMachine/src/solve.cpp > CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.i

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lezi/ClionProjects/ksparse_singleMachine/src/solve.cpp -o CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.s

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.requires:

.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.requires

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.provides: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.requires
	$(MAKE) -f CMakeFiles/sparse_opt_parallel_single_machine.dir/build.make CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.provides.build
.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.provides

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.provides.build: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o


CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o: CMakeFiles/sparse_opt_parallel_single_machine.dir/flags.make
CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o: /home/lezi/ClionProjects/ksparse_singleMachine/src/topK.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o -c /home/lezi/ClionProjects/ksparse_singleMachine/src/topK.cpp

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lezi/ClionProjects/ksparse_singleMachine/src/topK.cpp > CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.i

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lezi/ClionProjects/ksparse_singleMachine/src/topK.cpp -o CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.s

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.requires:

.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.requires

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.provides: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.requires
	$(MAKE) -f CMakeFiles/sparse_opt_parallel_single_machine.dir/build.make CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.provides.build
.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.provides

CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.provides.build: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o


# Object files for target sparse_opt_parallel_single_machine
sparse_opt_parallel_single_machine_OBJECTS = \
"CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o" \
"CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o" \
"CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o" \
"CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o"

# External object files for target sparse_opt_parallel_single_machine
sparse_opt_parallel_single_machine_EXTERNAL_OBJECTS =

sparse_opt_parallel_single_machine: CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o
sparse_opt_parallel_single_machine: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o
sparse_opt_parallel_single_machine: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o
sparse_opt_parallel_single_machine: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o
sparse_opt_parallel_single_machine: CMakeFiles/sparse_opt_parallel_single_machine.dir/build.make
sparse_opt_parallel_single_machine: CMakeFiles/sparse_opt_parallel_single_machine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable sparse_opt_parallel_single_machine"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sparse_opt_parallel_single_machine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sparse_opt_parallel_single_machine.dir/build: sparse_opt_parallel_single_machine

.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/build

CMakeFiles/sparse_opt_parallel_single_machine.dir/requires: CMakeFiles/sparse_opt_parallel_single_machine.dir/main.cpp.o.requires
CMakeFiles/sparse_opt_parallel_single_machine.dir/requires: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/common.cpp.o.requires
CMakeFiles/sparse_opt_parallel_single_machine.dir/requires: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/solve.cpp.o.requires
CMakeFiles/sparse_opt_parallel_single_machine.dir/requires: CMakeFiles/sparse_opt_parallel_single_machine.dir/src/topK.cpp.o.requires

.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/requires

CMakeFiles/sparse_opt_parallel_single_machine.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sparse_opt_parallel_single_machine.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/clean

CMakeFiles/sparse_opt_parallel_single_machine.dir/depend:
	cd /home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lezi/ClionProjects/ksparse_singleMachine /home/lezi/ClionProjects/ksparse_singleMachine /home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release /home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release /home/lezi/.CLion2016.2/system/cmake/generated/ksparse_singleMachine-ba306e0f/ba306e0f/Release/CMakeFiles/sparse_opt_parallel_single_machine.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sparse_opt_parallel_single_machine.dir/depend

