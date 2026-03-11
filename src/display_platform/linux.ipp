#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>

#include "display.h"

const char* CURSOR_HIDE = "\033[?25l";
const char* CURSOR_SHOW = "\033[?25h";

void Display::init() {
    std::cout << CURSOR_HIDE;
}

void Display::drop() {
    std::cout << CURSOR_SHOW;
    //std::cout << RESET << std::endl;
}

int Display::get_terminal_width() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        return 80;
    }
    return w.ws_col;
}

int Display::get_terminal_height() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        return 80;
    }
    return w.ws_row;
}
