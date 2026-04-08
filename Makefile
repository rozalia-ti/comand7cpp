CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -pedantic -g
LDFLAGS :=

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

INCLUDES := -I$(INC_DIR)

TARGET := plusi
EXECUTABLE := $(BIN_DIR)/$(TARGET)

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all
all: dirs $(EXECUTABLE)

.PHONY: dirs
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: rebuild
rebuild: clean all

.PHONY: run
run: all
	./$(EXECUTABLE)

.PHONY: debug
debug: all
	gdb ./$(EXECUTABLE)
