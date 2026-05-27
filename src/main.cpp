#include "point.h"
#include "display.h"
#include "distribution.h"
#include "ai.h"

#include <unordered_map>
#include <algorithm>

const int FPS = 60;
const size_t NUM_POINTS = 100;
const size_t L0 = 128 * 4;
const size_t L1 = 64 * 4;
const size_t L2 = 32 * 4;
const size_t L3 = 2;

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

int main_() {
    std::ranlux48 engine(std::random_device{}());
    std::uniform_real_distribution<double> position_distribution(0.0, 100.0);
    //std::uniform_int_distribution<uint8_t> class_distribution(0, 1);
    std::discrete_distribution<uint8_t> class_distribution({3, 1});
    std::uniform_real_distribution<double> equation_distribution(-3.0, 3.0);

    // std::vector<Point> points = generate_random(engine, NUM_POINTS, position_distribution, class_distribution);

    auto linear = [](double x) {
        return x;
    };

    auto linear_df = [](double _x) {
        return 1.0;
    };

    auto relu = [](double x) {
        return x > 0 ? x : x * 0.01;
    };

    auto relu_df = [](double x) {
        return x > 0 ? 1.0 : 0.01;
    };

    AiMatrix ai;
    ai.configure({
        { L0, 0, relu, relu_df },
        { L1, 1, relu, relu_df },
        { L2, 1, relu, relu_df },
        { L3, 1, linear, linear_df }
    });
    ai.randomize();

    for (int m = 0; m < 10000; m++) {
        auto points_result = generate_linear(engine, NUM_POINTS, position_distribution, equation_distribution);
        auto points = points_result.points;

        std::vector<double> ai_input(L0, 0);

        for (size_t i = 0; i < points.size(); i++) {
            ai_input[i * 4 + 0] = points[i].x / 100.0;
            ai_input[i * 4 + 1] = points[i].y / 100.0;
            ai_input[i * 4 + 2] = (double)points[i].cls;
            ai_input[i * 4 + 3] = 1.0;
        }

        std::vector<double> ai_target = {points_result.k, points_result.b};

        for (int n = 0; n < 10; n++) {
            ai.train(ai_input, ai_target, 0.00001);
        }

        //test
        //
        //
        double error = 0.0;

        for (int x = 0; x < 25; x++) {
            points_result = generate_linear(engine, NUM_POINTS, position_distribution, equation_distribution);
            points = points_result.points;

            ai_input = std::vector<double>(L0, 0);

            for (size_t i = 0; i < points.size(); i++) {
                ai_input[i * 4 + 0] = points[i].x / 100.0;
                ai_input[i * 4 + 1] = points[i].y / 100.0;
                ai_input[i * 4 + 2] = (double)points[i].cls;
                ai_input[i * 4 + 3] = 1.0;
            }

            const auto ai_output = ai.process(ai_input);

            double k = ai_output[0];
            double b = ai_output[1];
            error += sqrtl((b - points_result.b) * (b - points_result.b) + (k - points_result.k) * (k - points_result.k));

            std::cout << m << " - " << k << " " << b << " vs " << points_result.k << " " << points_result.b << std::endl;
        }

        std::cout << m << " - " << error / 100.0 << std::endl;
    }

    Display display(FPS, 101, 101);

    display.mainloop([&](DisplayCtx& ctx){
        ctx.clear();

        double x1 = -256.0;
        double y1 = x1 * k + b;
        double x2 = 256.0;
        double y2 = x2 * k + b;

        ctx.draw_line((int)x1, (int)y1, (int)x2, (int)y2, 4);

        for (auto& p : points) {
            if (p.cls == 0) ctx.draw_point((int)p.x, (int)p.y, 1);
            if (p.cls == 1) ctx.draw_point((int)p.x, (int)p.y, 2);
        }
        return ShouldExit::N;
    });

    return 0;
}
