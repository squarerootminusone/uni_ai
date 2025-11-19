#pragma once
#include "grading_helpers.h"
#include "your_code_here.h"
// Suppress warnings in third-party code.
#include <framework/disable_all_warnings.h>


TEST_CASE("ImageScale")
{
    SECTION("Copy")
    {
        auto rnd = RandomGenerator(0.0f, 1.0f);
        auto src_image = randomImageRGB(64, 57, rnd);
        auto src_copy = ImageRGB(src_image);
        
        CAPTURE(glm::to_string(src_copy.data[0])); // Will be reported in the error message


        // Call user method
        auto multiplier = 0.2f;
        auto dst_image = scaleImageCopy(src_image, multiplier);

        // Check the input has not been modified.
        CHECK_GLM(src_image.data[0], src_copy.data[0]);
        CHECK(calcImageRMSE(src_image, src_copy) == APPROX_FLOAT(0.0f));

        // Check the output has been modified.
        for (auto i = 0; i < dst_image.data.size(); i++) {
            REQUIRE(dst_image.data[i].x == APPROX_FLOAT(src_image.data[i].x * multiplier));
        }
    }

    SECTION("InPlace")
    {
        auto rnd = RandomGenerator(0.0f, 1.0f);
        auto src_image = randomImageRGB(64, 57, rnd);
        auto src_copy = ImageRGB(src_image);

        CAPTURE(glm::to_string(src_copy.data[0])); // Will be reported in the error message

        // Call user method
        auto multiplier = 0.55f;
        scaleImageInPlace(src_image, multiplier);

        // Check the input has been modified.
        CHECK(calcImageRMSE(src_image, src_copy) != APPROX_FLOAT(0.0f));

        // Check the input is correct.
        auto gt = ImageRGB(src_copy);
        std::transform(gt.data.begin(), gt.data.end(), gt.data.begin(), [&](auto x) { return x * multiplier; });
        CHECK(calcImageRMSE(src_image, gt) == APPROX_FLOAT(0.0f));
    }
}


TEST_CASE("Parallel")
{
    SECTION("MinVal")
    {
        auto res = true;
        for (int i = 0; i < 10; i++) {
            auto rnd = RandomGenerator(0.0f, 1.0f);
            auto data = randomVectorFloat(400, rnd);

            // Call user method
            auto answer = getMinimumValue(data);

            // Refence.
            auto gt = std::reduce(data.begin(), data.end(), data[0], [](auto a, auto b) { return std::min(a, b); });

            if (answer != APPROX_FLOAT(gt)) {
                CHECK(answer == APPROX_FLOAT(gt));
                res = false;
                break;
            }

        }
        
        if (res) {
            CHECK(true);
        }
           
    }

    SECTION("Sum")
    {
        auto res = true;
        for (int i = 0; i < 10; i++) {
            auto rnd = RandomGenerator(0.0f, 1.0f);
            auto data = randomVectorFloat(400, rnd);

            // Call user method
            auto answer = getSum(data);

            // Refence.
            auto gt = std::reduce(data.begin(), data.end(), 0.0f, [](auto a, auto b) { return a + b; });

            if (answer != APPROX_FLOAT(gt)) {
                CHECK(answer == APPROX_FLOAT(gt));
                res = false;
                break;
            }
        }
       
        if (res) {
            CHECK(true);
        }
    }
}
