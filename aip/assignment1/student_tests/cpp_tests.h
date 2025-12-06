#pragma once
#include "grading_helpers.h"
#include "your_code_here.h"
// Suppress warnings in third-party code.
#include <framework/disable_all_warnings.h>
#include <filesystem>
DISABLE_WARNINGS_PUSH()
#include <glm/gtx/string_cast.hpp>
#include <catch2/catch_all.hpp>
DISABLE_WARNINGS_POP()

static const std::filesystem::path dataDirPath { DATA_DIR };
static const std::filesystem::path rawDataDirPath { dataDirPath / "raw_data" };

// Set up images used for running the tests
const auto memorial_img = ImageRGB(dataDirPath / "memorial2_half.hdr");
const auto memorial_img_source = ImageRGB(dataDirPath / "cat.png");
const auto memorial_img_mask = ImageRGB(dataDirPath / "cat_mask.png");

const auto kitchen_img = ImageRGB(dataDirPath / "kitchen_probe.hdr");

const auto plane_target_img = ImageRGB(dataDirPath / "plane_target.jpg");
const auto plane_src_img = ImageRGB(dataDirPath / "plane_src.jpg");
const auto plane_mask_img = ImageRGB(dataDirPath / "plane_mask.png");

const auto small_img_4x3 = ImageRGB(dataDirPath / "small_rnd_img_4x3.png");


////////////////////////////////////////////////////
// 0.GetRGBImageMinMax
////////////////////////////////////////////////////
void checkGetRGBImageMinMax(const ImageRGB& image, const std::string& id = "") {
    glm::vec2 output = loadVecFromFile<glm::vec2>(rawDataDirPath / ("checkGetRGBImageMinMax_" + id + "_output.bin"));

    try {
        auto user_copy_image = ImageRGB(image);
        auto user_output = getRGBImageMinMax(user_copy_image);
        auto result = output.x == user_output.x && output.y == user_output.y;
        CHECK(result);
    } catch (...) {
        CHECK(false);
    }

};

TEST_CASE("getRGBImageMinMax") {   
    SECTION("MemorialImage") {
        checkGetRGBImageMinMax(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage") {
        checkGetRGBImageMinMax(kitchen_img, "KitchenImage");
    }
}

////////////////////////////////////////////////////
// 1.NormalizeRGBImage
////////////////////////////////////////////////////
void checkNormalizeRGBImage(const ImageRGB& image, const std::string& id = ""){
    auto output = ImageRGB();
    output.readBinary(rawDataDirPath / ("checkNormalizeRGBImage_" + id + "_output.bin"));

    auto user_copy_image = ImageRGB(image);
    auto user_output = normalizeRGBImage(user_copy_image);

    CHECK(calcImageRMSE(output, user_output) == APPROX_FLOAT(0.0f));
};

TEST_CASE("NormalizeRGBImage")
{   
    SECTION("MemorialImage") {
        checkNormalizeRGBImage(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage") {
        checkNormalizeRGBImage(kitchen_img, "KitchenImage");
    }

}

////////////////////////////////////////////////////
// 2.ApplyGamma
////////////////////////////////////////////////////
void checkApplyGamma(const ImageRGB& image, const std::string& id = "", float gamma = 1 / 3.3f) {
    auto output = ImageRGB();
    output.readBinary(rawDataDirPath / ("checkApplyGamma_" + id + "_output.bin"));

    try {
        auto user_copy_image = ImageRGB(image);
        auto user_output = applyGamma(user_copy_image, gamma);

        CHECK(calcImageRMSE(output, user_output) == APPROX_FLOAT(0.0f));
    } catch (...) {
        CHECK(false);
    }

};

TEST_CASE("ApplyGamma")
{
    SECTION("MemorialImage") {
        checkApplyGamma(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage") {
        checkApplyGamma(kitchen_img, "KitchenImage");
    }
}

////////////////////////////////////////////////////
// 3.rgbToLuminance
////////////////////////////////////////////////////
void checkrgbToLuminance(const ImageRGB& image, const std::string& id = "") {
    auto output = ImageFloat();
    output.readBinary(rawDataDirPath / ("checkrgbToLuminance_" + id + "_output.bin"));

    try {
        auto user_copy_image = ImageRGB(image);
        auto user_output = rgbToLuminance(user_copy_image);
        CHECK(calcImageRMSE(output, user_output) == APPROX_FLOAT(0.0f));
    } catch (...) {
        CHECK(false);
    }

};

TEST_CASE("rgbToLuminance")
{
    SECTION("MemorialImage") {
        checkrgbToLuminance(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage") {
        checkrgbToLuminance(kitchen_img, "KitchenImage");
    }
}

////////////////////////////////////////////////////
// 4.bilateralFilter
////////////////////////////////////////////////////
void checkBilateralFilter(const ImageRGB& image, const int size, const float space_sigma, const float range_sigma, const std::string& id = "")
{
    auto log_lum_H = ImageFloat();
    log_lum_H.readBinary(rawDataDirPath / ("checkBilateralFilter_" + id + "_log_lum_H.bin"));

    auto reference_output = ImageFloat();
    reference_output.readBinary(rawDataDirPath / ("checkBilateralFilter_" + id + "_output.bin"));

    auto user_output = bilateralFilter(log_lum_H, size, space_sigma, range_sigma);

    CHECK(calcImageRMSE(reference_output, user_output) == APPROX_FLOAT(0.0f));
}

TEST_CASE("bilateralFilter")
{
    SECTION("MemorialImage")
    {
        const int filter_size = 27; // must be an odd integer
        const float space_sigma = filter_size / 6.4f;
        const float range_sigma = 1.0f;

        checkBilateralFilter(memorial_img, filter_size, space_sigma, range_sigma, "MemorialImage");
    }

    SECTION("KitchenImage")
    {
        const int filter_size = 9; // must be an odd integer
        const float space_sigma = filter_size / 6.4f;
        const float range_sigma = 0.5f;

        checkBilateralFilter(kitchen_img, filter_size, space_sigma, range_sigma, "KitchenImage");
    }

}

////////////////////////////////////////////////////
// 5.applyDurandToneMappingOperator
////////////////////////////////////////////////////
void checkApplyDurandToneMappingOperator(const ImageRGB& image, const float base_scale, const float output_gain, const std::string& id = "")
{
    const int filter_size = 27; // must be an odd integer
    const float space_sigma = filter_size / 6.4f;
    const float range_sigma = 1.0f;

    auto base_image = ImageFloat();
    auto detail_image = ImageFloat();
    auto reference_output = ImageFloat();
    base_image.readBinary(rawDataDirPath / ("checkApplyDurandToneMappingOperator_" + id + "_base_image.bin"));
    detail_image.readBinary(rawDataDirPath / ("checkApplyDurandToneMappingOperator_" + id + "_detail_image.bin"));
    reference_output.readBinary(rawDataDirPath / ("checkApplyDurandToneMappingOperator_" + id + "_output.bin"));

    auto user_output = applyDurandToneMappingOperator(base_image, detail_image, base_scale, output_gain);

    CHECK(calcImageRMSE(reference_output, user_output) == APPROX_FLOAT(0.0f));
}

TEST_CASE("applyDurandToneMappingOperator")
{
    SECTION("MemorialImage")
    {
        const float base_scale = 0.15f;
        const float output_gain = 0.5f;

        checkApplyDurandToneMappingOperator(memorial_img, base_scale, output_gain, "MemorialImage");
    }

    SECTION("KitchenImage")
    {
        const float base_scale = 0.5f;
        const float output_gain = 0.15f;

        checkApplyDurandToneMappingOperator(kitchen_img, base_scale, output_gain, "KitchenImage");
    }

}

////////////////////////////////////////////////////
// 6.rescaleRgbByLuminance
////////////////////////////////////////////////////
void checkRescaleRGBByLuminance(const ImageRGB& image, const std::string& id = "")
{

    auto tmo_luminance = ImageFloat();
    auto hdr_luminance = ImageFloat();
    auto reference_output = ImageRGB();
    tmo_luminance.readBinary(rawDataDirPath / ("checkRescaleRGBByLuminance_" + id + "_tmo_luminance.bin"));
    hdr_luminance.readBinary(rawDataDirPath / ("checkRescaleRGBByLuminance_" + id + "_hdr_luminance.bin"));
    reference_output.readBinary(rawDataDirPath / ("checkRescaleRGBByLuminance_" + id + "_output.bin"));

    auto user_output = rescaleRgbByLuminance(image, hdr_luminance, tmo_luminance);

    CHECK(calcImageRMSE(reference_output, user_output) == APPROX_FLOAT(0.0f));
};

TEST_CASE("rescaleRGBByLuminance")
{
    SECTION("MemorialImage")
    {
        checkRescaleRGBByLuminance(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage")
    {
        checkRescaleRGBByLuminance(kitchen_img, "KitchenImage");
    }

}

////////////////////////////////////////////////////
// 7.rescaleRgbByLuminance
////////////////////////////////////////////////////
void checkGetGradients(const ImageRGB& image, const std::string& id = "") {

    auto hdr_luminance = ImageFloat();
    hdr_luminance.readBinary(rawDataDirPath / ("checkGetGradients_" + id + "_hdr_luminance.bin"));

    auto output = ImageGradient();
    output.readBinary(rawDataDirPath / ("checkGetGradients_" + id + "_output"));
    
    try {
        auto user_output = getGradients(hdr_luminance);
        auto rmse = calcGradientRMSE(output, user_output);

        CHECK(rmse == APPROX_FLOAT(0.0f));
        
    } catch (...) {
        CHECK(false);
    }
    
};

TEST_CASE("getGradients")
{   
    SECTION("MemorialImage") {
        checkGetGradients(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage") {
        checkGetGradients(kitchen_img, "KitchenImage");
    }

    SECTION("4x3Image") {
        checkGetGradients(small_img_4x3, "4x3Image");
    }
    
}

////////////////////////////////////////////////////
// 8.copySourceGradientsToTargetXYZ
////////////////////////////////////////////////////
void checkCopySourceGradientsToTarget(const ImageRGB& source_image, const ImageRGB& target_image, const ImageRGB& mask_image, const std::string& id = "")
{   
    ImageFloat source_mask = imageRgbToFloat(mask_image);
    auto source_gradients = ImageGradient();
    auto target_gradients = ImageGradient();
    auto output_luminance = ImageGradient();
    source_gradients.readBinary(rawDataDirPath / ("checkCopySourceGradientsToTarget_" + id + "_source_gradients"));
    target_gradients.readBinary(rawDataDirPath / ("checkCopySourceGradientsToTarget_" + id + "_target_gradients"));
    output_luminance.readBinary(rawDataDirPath / ("checkCopySourceGradientsToTarget_" + id + "_output_luminance.bin"));

    try {
        auto user_output = copySourceGradientsToTarget(source_gradients, target_gradients, source_mask);

        float rmse = calcGradientRMSE(output_luminance, user_output);
        CHECK(rmse == APPROX_FLOAT(0.0f));
    } catch (...) {
        CHECK(false);
    }
};


TEST_CASE("copySourceGradientsToTarget")
{
    SECTION("MemorialImage")
    {
        checkCopySourceGradientsToTarget(memorial_img_source, memorial_img, memorial_img_mask, "MemorialImage");
    }

    SECTION("PlaneImage")
    {
        checkCopySourceGradientsToTarget(plane_src_img, plane_target_img, plane_mask_img, "PlaneImage");
    }

}

////////////////////////////////////////////////////
// 9.getDivergenceXYZ
////////////////////////////////////////////////////
void checkGetDivergence(const ImageRGB& target_image, const std::string& id = "")
{
    auto target_gradients = ImageGradient();
    target_gradients.readBinary(rawDataDirPath / ("checkGetDivergence_" + id + "_target_gradients.bin"));

    auto output = ImageFloat();
    output.readBinary(rawDataDirPath / ("checkGetDivergence_" + id + "_output.bin"));

    try {
        auto user_output = getDivergence(target_gradients);
        float rmse = calcImageRMSE(output, user_output);

        CHECK(rmse == APPROX_FLOAT(0.0f));
        
    } catch (...) {
        CHECK(false);
    }
};

TEST_CASE("getDivergence")
{
    SECTION("MemorialImage")
    {
        checkGetDivergence(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage")
    {
        checkGetDivergence(kitchen_img, "KitchenImage");
    }

    SECTION("4x3Image")
    {
        checkGetDivergence(small_img_4x3, "4x3Image");
    }
}

////////////////////////////////////////////////////
// 10.solvePoissonXYZ
////////////////////////////////////////////////////
void checkSolvePoisson(const ImageRGB& target_image, const std::string& id = "")
{
    auto target_image_luminance = rgbToLuminance(target_image);

    auto divergence = ImageFloat();
    divergence.readBinary(rawDataDirPath / ("checkSolvePoisson_" + id + "_divergence.bin"));

    auto emptyImage = ImageFloat(target_image_luminance.width, target_image_luminance.height);
    auto output = ImageFloat();
    output.readBinary(rawDataDirPath / ("checkSolvePoisson_" + id + "_output.bin"));

    try {
        auto user_output = solvePoisson(emptyImage, divergence);

        auto rmse = calcImageRMSE(output, user_output);
        CHECK(rmse == APPROX_FLOAT(0.0f));
    } catch (...) {
        CHECK(false);
    }
};

TEST_CASE("solvePoisson")
{
    SECTION("MemorialImage")
    {
        checkSolvePoisson(memorial_img, "MemorialImage");
    }

    SECTION("KitchenImage")
    {
        checkSolvePoisson(kitchen_img, "KitchenImage");
    }

    SECTION("4x3Image")
    {
        checkSolvePoisson(small_img_4x3, "4x3Image");
    }
}