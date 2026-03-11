#ifndef DISPLAY_H
#define DISPLAY_H

#include <vector>
#include <functional>

extern const char* BLOCK;
extern const char* RESET;
extern const char* RED;
extern const char* WHITE;

enum ShouldExit {
    Y,
    N
};

class Display;

class DisplayGrid {
private:
    int* buffer;
    int width;
    int height;

public:
    DisplayGrid();
    int get_width();
    int get_height();
    void resize(int new_width, int new_height);
    int* block_at(int x, int y);
};

class DrawCommand {
public:
    virtual void apply(DisplayGrid& grid);
    virtual ~DrawCommand();
};

class DrawPoint : public DrawCommand {
private:
    int pos_x;
    int pos_y;
    int color;

public:
    DrawPoint(int _pos_x, int _pos_y, int _color);

    void apply(DisplayGrid& grid) override;
};

class DrawLine : public DrawCommand {
private:
    int pos1_x;
    int pos1_y;
    int pos2_x;
    int pos2_y;
    int color;

public:
    DrawLine(int _pos1_x, int _pos1_y, int _pos2_x, int _pos2_y, int _color);

    void apply(DisplayGrid& grid) override;
};

class DrawClear : public DrawCommand {
public:
    DrawClear();

    void apply(DisplayGrid& grid) override;
};

class DisplayCtx {
private:
    Display* display;
    std::vector<DrawCommand*> draw_commands;

public:
    DisplayCtx(Display* _display);

    void clear();
    void draw_point(int pos_x, int pos_y, int color);
    void draw_line(int pos1_x, int pos1_y, int pos2_x, int pos2_y, int color);

    void flush_draw_commands();
};

class Display {
private:
    int fps;
    bool needs_redraw;
    int terminal_width;
    int terminal_height;
    int width;
    int height;

    void init();
    void drop();

    static int get_terminal_width();
    static int get_terminal_height();

public:
    DisplayGrid grid;

    Display(int _fps, int _width, int _height);
    ~Display();

    void mainloop(std::function<ShouldExit(DisplayCtx&)> update_fn);
    void draw_grid();
};

#endif
