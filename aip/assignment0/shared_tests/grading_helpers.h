#pragma once
#include "your_code_here.h"
// Suppress warnings in third-party code.
//#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <catch2/catch_all.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/matrix_transform_2d.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()
#include <random>
#include <tuple>

#include <framework/image.h>

using ImageFloat = Image<float>;
using ImageRGB = Image<glm::vec3>;

static constexpr size_t randomSeed = 12345;

class RandomGenerator {
public:
    inline RandomGenerator(float lowest = 0.0f, float highest = 1.0f)
        : m_floatDist(lowest, highest) {};

    inline float nextFloat()
    {
        return m_floatDist(m_randomEngine);
    }

    inline glm::vec3 nextVec3()
    {
        return glm::vec3(nextFloat(), nextFloat(), nextFloat());
    }

private:
    // Known engine + fixed seed = reproducible tests.
    std::mt19937 m_randomEngine { randomSeed };
    std::uniform_real_distribution<float> m_floatDist;
};

ImageFloat randomImageFloat(const int width, const int height, RandomGenerator& rnd)
{
    auto image = ImageFloat(width, height);
    std::generate(image.data.begin(), image.data.end(), [&]() { return rnd.nextFloat(); });
    return image;
}

ImageRGB randomImageRGB(const int width, const int height, RandomGenerator& rnd)
{
    auto image = ImageRGB(width, height);
    std::generate(image.data.begin(), image.data.end(), [&]() { return rnd.nextVec3(); });
    return image;
}

std::vector<float> randomVectorFloat(const int size, RandomGenerator& rnd) {
    auto data = std::vector<float>(size);
    std::generate(data.begin(), data.end(), [&]() { return rnd.nextFloat(); });
    return data;
}


#define APPROX_FLOAT(x) Catch::Approx(x).margin(1e-5f)

#define CHECK_GLM(x, y) CHECK(glm::length((x) - (y)) == APPROX_FLOAT(0))
#define REQUIRE_GLM(x, y) REQUIRE(glm::length((x) - (y)) == APPROX_FLOAT(0))

// ApproxZero by default uses an epsilon of 0.0f and for some reason it makes one of the tests fail ( 0.0f == Approx(0.0) fails... ).
// https://github.com/catchorg/Catch2/issues/1444
//
// https://stackoverflow.com/questions/56466022/what-is-the-canonical-way-to-check-for-approximate-zeros-in-catch2
#define ApproxZero Catch::Approx(0.0f).margin(1e-5f)


/// <summary>
/// https://stackoverflow.com/questions/48955718/c-how-to-calculate-rmse-between-2-vectors
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="a"></param>
/// <param name="b"></param>
/// <returns></returns>
template <typename T>
auto calcVectorRMSE(const std::vector<T>& a, const std::vector<T>& b)
{
    auto squareError = [](T a, T b) {
        auto e = a - b;
        return glm::dot(e, e);
    };

    auto sum = std::transform_reduce(a.begin(), a.end(), b.begin(), 0.0f, std::plus<>(), squareError);
    auto rmse = std::sqrt(sum / a.size());
    return rmse;
}

template <typename T>
auto calcImageRMSE(const T& a, const T& b)
{
    return calcVectorRMSE(a.data, b.data);
}