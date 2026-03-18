#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>
#include <string>

#include "display.h"

const std::string CURSOR_HIDE = "\033[?25l";
const std::string CURSOR_SHOW = "\033[?25h";

void Display::init() {
    std::cout << CURSOR_HIDE;
}

void Display::drop() {
    std::cout << CURSOR_SHOW;
    //std::cout << RESET << std::endl;
}

void Display::get_terminal_size(int& width, int& height) {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        width = 80;
        height = 80;
        return;
    }
    width = w.ws_col;
    height = w.ws_row;
}
