#include <thread>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

#include "display.h"

DisplayGrid::DisplayGrid() :
    buffer(nullptr),
    width(0),
    height(0)
{}

int DisplayGrid::get_width() {
    return width;
}

int DisplayGrid::get_height() {
    return height;
}

void DisplayGrid::resize(int new_width, int new_height) {
    if (width == new_width && height == new_height) {
        return;
    }

    width = new_width;
    height = new_height;

    if (buffer != nullptr) {
        delete buffer;
    }

    buffer = new int[width * height];
    for (int i = 0; i < width * height; i++) {
        buffer[i] = 0;
    }
}

int* DisplayGrid::block_at(int x, int y) {
    return &buffer[y * width + x];
}

DrawCommand::~DrawCommand() = default;

void DrawCommand::apply(DisplayGrid& grid) {
    (void)grid;
}

DrawPoint::DrawPoint(int _pos_x, int _pos_y, int _color) :
    pos_x(_pos_x),
    pos_y(_pos_y),
    color(_color)
{}

void DrawPoint::apply(DisplayGrid& grid) {
    if (pos_x < 0 || pos_y < 0 || pos_x >= grid.get_width() || pos_y >= grid.get_height()) {
        return;
    }

    auto block = grid.block_at(pos_x, pos_y);
    *block = color;
}

DrawLine::DrawLine(int _pos1_x, int _pos1_y, int _pos2_x, int _pos2_y, int _color) :
    pos1_x(_pos1_x),
    pos1_y(_pos1_y),
    pos2_x(_pos2_x),
    pos2_y(_pos2_y),
    color(_color)
{}

void DrawLine::apply(DisplayGrid& grid) {
    int x = pos1_x;
    int y = pos1_y;
    int dx = abs(pos2_x - pos1_x);
    int dy = abs(pos2_y - pos1_y);
    int sx = (pos1_x < pos2_x) ? 1 : -1;
    int sy = (pos1_y < pos2_y) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        if (x >= 0 && y >= 0 && x < grid.get_width() && y < grid.get_height()) {
            auto block = grid.block_at(x, y);
            *block = color;
        }

        if (x == pos2_x && y == pos2_y) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

DrawClear::DrawClear() {}

void DrawClear::apply(DisplayGrid& grid) {
    for (int x = 0; x < grid.get_width(); x++) {
        for (int y = 0; y < grid.get_height(); y++) {
            *grid.block_at(x, y) = 0;
        }
    }
}

DisplayCtx::DisplayCtx(Display* _display) :
    display(_display),
    draw_commands({})
{}

void DisplayCtx::clear() {
    this->draw_commands.push_back(new DrawClear());
}

void DisplayCtx::draw_point(int pos_x, int pos_y, int color) {
    this->draw_commands.push_back(new DrawPoint(pos_x, pos_y, color));
}

void DisplayCtx::draw_line(int pos1_x, int pos1_y, int pos2_x, int pos2_y, int color) {
    this->draw_commands.push_back(new DrawLine(pos1_x, pos1_y, pos2_x, pos2_y, color));
}

void DisplayCtx::flush_draw_commands() {
    for (DrawCommand* cmd : draw_commands) {
        cmd->apply(display->grid);
        delete cmd;
    }
    draw_commands.clear();
}

Display::Display(int _fps, int _width, int _height) :
    fps(_fps),
    needs_redraw(true),
    terminal_width(Display::get_terminal_width()),
    terminal_height(Display::get_terminal_height()),
    width(_width),
    height(_height)

{
    init();
    grid.resize(width, height);
}

Display::~Display() {
    drop();
}

void Display::mainloop(std::function<ShouldExit(DisplayCtx&)> update_fn) {
    while (true) {
        terminal_width = Display::get_terminal_width();
        terminal_height = Display::get_terminal_height() * 2;

        DisplayCtx ctx(this);
        auto should_exit = update_fn(ctx);

        ctx.flush_draw_commands();
        draw_grid();

        if (should_exit == ShouldExit::Y) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));
    }
}

void Display::draw_grid() {
    std::string buffer;
    buffer.reserve(grid.get_width() * grid.get_height() * 30);

    for (int y = (grid.get_height() - 1) / 2 * 2; y >= 0; y -= 2) {
        for (int x = 0; x < grid.get_width(); x++) {
            auto block_bottom = *grid.block_at(x, y);
            auto block_top = y + 1 < grid.get_height() ? *grid.block_at(x, y + 1) : 0;

            buffer += "\033[38;5;" + std::to_string(block_top) + "m" +
                      "\033[48;5;" + std::to_string(block_bottom) + "m" +
                      "\u2580";
        }
        buffer += "\033[0m\n";
    }

    buffer += "\033[38;5;0m\033[48;5;0m";
    for (int y = terminal_height - grid.get_height(); y >= 8; y -= 2) {
        for (int x = 0; x < grid.get_width(); x++) {
            buffer += "\u2580";
        }
        buffer += "\n";
    }

    buffer += "\033[1;1H";

    std::cout << buffer;
}

#ifdef _WIN32
#include "display_platform/windows.ipp"
#else
#include "display_platform/linux.ipp"
#endif
