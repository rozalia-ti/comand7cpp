CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -pedantic -g -fPIC -m64

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

PYBIND11_INCLUDES := $(shell py -m pybind11 --includes)
PYTHON_SUFFIX := $(shell py -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

PYTHON_ROOT := $(shell py -c "import sys; print(sys.base_prefix)")
PYTHON_VERSION := $(shell py -c "import sys; print(f'{sys.version_info[0]}{sys.version_info[1]}')")

PYTHON_LDFLAGS := -L$(PYTHON_ROOT) -lpython$(PYTHON_VERSION) -lkernel32 -luser32

LDFLAGS := -shared -m64 -static-libgcc -static-libstdc++ -Wl,-Bstatic -lpthread -lwinpthread -Wl,-Bdynamic $(PYTHON_LDFLAGS)
INCLUDES := -I$(INC_DIR) $(PYBIND11_INCLUDES)

TARGET := comand7
LIBRARY := $(BIN_DIR)/$(TARGET)$(PYTHON_SUFFIX)

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
LIB_SRCS := $(filter-out $(SRC_DIR)/main.cpp, $(SRCS))
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))
DEPS := $(OBJS:.o=.d)

ifeq ($(OS),Windows_NT)
    MKDIR = if not exist $(subst /,\\,$(1)) mkdir $(subst /,\\,$(1))
    RMDIR = if exist $(subst /,\\,$(1)) rmdir /s /q $(subst /,\\,$(1))
else
    MKDIR = mkdir -p $(1)
    RMDIR = rm -rf $(1)
endif

.PHONY: all
all: dirs $(LIBRARY)

.PHONY: dirs
dirs:
	@$(call MKDIR,$(OBJ_DIR))
	@$(call MKDIR,$(BIN_DIR))

$(LIBRARY): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

-include $(DEPS)

.PHONY: clean
clean:
	@$(call RMDIR,$(BUILD_DIR))

.PHONY: rebuild
rebuild: clean all