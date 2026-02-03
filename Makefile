EXE = RayBox_library.so
NVCC = /usr/bin/nvcc
CPP = g++
CPP_FLAGS = -std=c++11 -O3
NVCC_FLAGS = -arch=sm_75 --compiler-options
PY_BIND=`python3 -m pybind11 --includes`
SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include
SRCS = $(wildcard $(SRC_DIR)/*.c*)
OBJS = $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(notdir $(basename $(SRCS)))))
CUDA_ROOT_DIR=/usr/lib/cuda
# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

# Link c++ and CUDA compiled object files to target executable:
$(EXE): $(OBJS)
	$(CPP) -shared $(CPP_FLAGS) $(OBJS) -o $@ -I./$(INC_DIR) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CPP) $(CPP_FLAGS) $(PY_BIND) -fPIC -c $< -o $@ -I./$(INC_DIR)

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(CPP) $(CPP_FLAGS) $(PY_BIND) -fPIC -c $< -o $@ -I./$(INC_DIR)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -fPIC $(PY_BIND) -c $< -o $@ $(NVCC_LIBS) -I./$(INC_DIR)
# 	@echo "variable contains $(SRCS)"

clean:
	rm -f $(OBJ_DIR)/*

veryclean: clean
	rm -f $(EXE)
