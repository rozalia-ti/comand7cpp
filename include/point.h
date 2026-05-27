#ifndef POINT_H
#define POINT_H

#include <cstdint>
#include <iostream>

struct Point {
    double x, y;
    uint8_t cls;

    std::ostream& operator<<(std::ostream& os);
};

#endif
