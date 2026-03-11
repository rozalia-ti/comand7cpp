#include "point.h"
#include "display.h"
#include "distribution.h"

#include <unordered_map>
#include <algorithm>

const int FPS = 60;
const size_t NUM_POINTS = 100;

int calc_score(std::vector<Point>& points, double k, double b) {
    int score = 0;

    std::unordered_map<uint8_t, int> side1_cls_cnt;
    std::unordered_map<uint8_t, int> side2_cls_cnt;

    for (auto& p : points) {
        auto y = k * p.x + b;
        if (y > p.y) {
            side1_cls_cnt[p.cls]++;
        } else {
            side2_cls_cnt[p.cls]++;
        }
    }

    uint8_t side1_cls_majority = 0;
    int side1_cls_majority_cnt = 0;
    for (const auto& pair : side1_cls_cnt) {
        if (pair.second > side1_cls_majority_cnt) {
            side1_cls_majority_cnt = pair.second;
            side1_cls_majority = pair.first;
        }
    }

    uint8_t side2_cls_majority = 0;
    int side2_cls_majority_cnt = 0;
    for (const auto& pair : side2_cls_cnt) {
        if (pair.second > side2_cls_majority_cnt) {
            side2_cls_majority_cnt = pair.second;
            side2_cls_majority = pair.first;
        }
    }

    if (side1_cls_majority == side2_cls_majority) {
        if (side1_cls_majority_cnt > side2_cls_majority_cnt) {
            side2_cls_majority = 1 - side1_cls_majority;
        } else {
            side1_cls_majority = 1 - side2_cls_majority;
        }
    }

    score += side1_cls_cnt[side1_cls_majority] + side2_cls_cnt[side2_cls_majority];

    return score;
}

int main() {
    std::ranlux48 engine(std::random_device{}());
    std::uniform_real_distribution<double> position_distribution(0.0, 100.0);
    //std::uniform_int_distribution<uint8_t> class_distribution(0, 1);
    std::discrete_distribution<uint8_t> class_distribution({3, 1});
    std::uniform_real_distribution<double> equation_distribution(-3.0, 3.0);

    std::vector<Point> points = generate_random(engine, NUM_POINTS, position_distribution, class_distribution);
    points = generate_linear(engine, NUM_POINTS, position_distribution, equation_distribution);

    Display display(FPS, 101, 101);

    double opti_k_best = 0.0;
    double opti_b_best = 0.0;

    display.mainloop([&](DisplayCtx& ctx){
        for (int i = 0; i < 500; i++) {
            std::uniform_real_distribution<double> dist(-100.0, 100.0);
            double opti_k_current = dist(engine);
            double opti_b_current = dist(engine);

            int score_best = calc_score(points, opti_k_best, opti_b_best);
            int score_current = calc_score(points, opti_k_current, opti_b_current);

            if (score_current > score_best) {
                opti_k_best = opti_k_current;
                opti_b_best = opti_b_current;
            }
        }

        ctx.clear();

        double x1 = -256.0;
        double y1 = x1 * opti_k_best + opti_b_best;
        double x2 = 256.0;
        double y2 = x2 * opti_k_best + opti_b_best;

        ctx.draw_line((int)x1, (int)y1, (int)x2, (int)y2, 4);

        for (auto& p : points) {
            if (p.cls == 0) ctx.draw_point((int)p.x, (int)p.y, 1);
            if (p.cls == 1) ctx.draw_point((int)p.x, (int)p.y, 2);
        }
        return ShouldExit::N;
    });

    return 0;
}
