CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -pedantic -g
LDFLAGS :=
PYTHON := python3

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

INCLUDES := -I$(INC_DIR)

TARGET := plusi
EXECUTABLE := $(BIN_DIR)/$(TARGET)
PY_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)
PY_INCLUDES := $(shell $(PYTHON)-config --includes)
PY_LDFLAGS := $(shell $(PYTHON)-config --ldflags)
PY_CXXFLAGS := -Wno-missing-field-initializers
PY_MODULE := cppnn$(PY_EXT_SUFFIX)
PY_BINDING_SRC := bindings/cppnn_module.cpp
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
PY_SHARED_FLAGS := -bundle -undefined dynamic_lookup
else
PY_SHARED_FLAGS := -shared
endif

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all
all: dirs $(EXECUTABLE)

.PHONY: python
python: $(PY_MODULE)

.PHONY: dirs
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(PY_MODULE): $(PY_BINDING_SRC) $(INC_DIR)/ai.h
	$(CXX) $(CXXFLAGS) $(PY_CXXFLAGS) $(INCLUDES) $(PY_INCLUDES) $(PY_SHARED_FLAGS) -fPIC -o $@ $(PY_BINDING_SRC) $(PY_LDFLAGS)

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) cppnn*.so cppnn*.so.dSYM

.PHONY: rebuild
rebuild: clean all

.PHONY: run
run: all
	./$(EXECUTABLE)

.PHONY: debug
debug: all
	gdb ./$(EXECUTABLE)
