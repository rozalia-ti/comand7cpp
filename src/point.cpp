#include <iostream>
#include "point.h"

std::ostream& Point::operator<<(std::ostream& os) {
    os << "(" << x << ", " << y << " - " << (int)cls << ")";
    return os;
}
