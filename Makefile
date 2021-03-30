INCLUDE_DIR := ./include
SRC_DIR := ./src
OBJ_DIR := ./obj

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
CUDA_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_FILES))

CXX := nvcc
CXX_FLAGS := -I$(INCLUDE_DIR)

kmeans: $(OBJ_FILES) $(CUDA_FILES)
	$(CXX) -o $@ $^ $(CXX_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c -o $@ $< $(CXX_FLAGS) 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CXX) -c -o $@ $< $(CXX_FLAGS) 