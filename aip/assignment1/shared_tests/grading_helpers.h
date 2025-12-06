#pragma once
#include "your_code_here.h"
// Suppress warnings in third-party code.
//#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <catch2/catch_all.hpp>
#define GLM_ENABLE_EXPERIMENTAL
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

#define APPROX_FLOAT_HALF(x) Catch::Approx(x).margin(0.005f)

#define APPROX_FLOAT(x) Catch::Approx(x).margin(1e-2f)

#define CHECK_GLM(x, y) CHECK(glm::length((x) - (y)) == APPROX_FLOAT(0))
#define REQUIRE_GLM(x, y) REQUIRE(glm::length((x) - (y)) == APPROX_FLOAT(0))

// ApproxZero by default uses an epsilon of 0.0f and for some reason it makes one of the tests fail ( 0.0f == Approx(0.0) fails... ).
// https://github.com/catchorg/Catch2/issues/1444
//
// https://stackoverflow.com/questions/56466022/what-is-the-canonical-way-to-check-for-approximate-zeros-in-catch2
#define ApproxZero Catch::Approx(0.0f).margin(1e-2f)


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

auto calcGradientRMSE(const ImageGradient& a, const ImageGradient& b) {
    return 0.5f * calcImageRMSE(a.dx, b.dx) + 0.5f * calcImageRMSE(a.dy, b.dy);
}

void saveVec2ToFile(const glm::vec2& vec, const std::filesystem::path& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&vec), sizeof(vec));
        file.close();
    }
}

void saveVec3ToFile(const glm::vec3& vec, const std::filesystem::path& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&vec), sizeof(vec));
        file.close();
    }
}

template<typename T>
void saveVecToFile(const T& vec, const std::filesystem::path& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(&vec), sizeof(vec));
        file.close();
    }
}

glm::vec2 loadVec2FromFile(const std::filesystem::path& filepath) {
    glm::vec2 vec;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&vec), sizeof(vec));
        file.close();
    }
    return vec;
}

glm::vec3 loadVec3FromFile(const std::filesystem::path& filepath) {
    glm::vec3 vec;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&vec), sizeof(vec));
        file.close();
    }
    return vec;
}

template<typename T>
T loadVecFromFile(const std::filesystem::path& filepath) {
    T vec;
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&vec), sizeof(vec));
        file.close();
    }
    return vec;
}
