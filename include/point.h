#ifndef POINT_H
#define POINT_H

#include <iostream>

struct Point {
    double x, y;
    uint8_t cls;

    friend std::ostream& operator<<(std::ostream& os, const Point& p);
};

std::ostream& operator<<(std::ostream& os, const Point& p);

#endif
