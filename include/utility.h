/*
 * @Author: juling julinger@qq.com
 * @Date: 2026-03-11 14:18:01
 * @LastEditors: juling julinger@qq.com
 * @LastEditTime: 2026-03-17 16:22:26
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <string>
#include <vector>

#include "defines.h"
/*
#################### CONVERSION UTILITY ####################
*/

// Convert a cv mat to a glm mat
bool fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat);
bool fromCV2GLM(const cv::Mat& cvmat, glm::mat3* glmmat);

// Convert a glm mat to a cv mat
bool fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat);
bool fromGLM2CV(const glm::mat3& glmmat, cv::Mat* cvmat);
bool fromGLM2CV(const glm::mat3& glmmat, cv::Matx33d* in_mat);

/*
#################### FILE UTILITY ####################
*/

/**
 * @brief Get all filenames in a directory
 *
 * @param[out] in_filePathVector File names
 * @param in_path Directory
 * @param in_extension File type to look for
 */
void filesInDirectory(std::vector<std::string>& in_filePathVector,
                      const std::string& in_path,
                      const std::string& in_extension);

/**
 * @brief Load the depth image from lm dataset
 *
 */
cv::Mat loadDepthLineModDataset(const std::string& a_name);

/**
 * @brief Read the settings yaml
 *
 * @param[out] in_camParams Camera Settings
 * @param[out] in_tempGenSettings Linemod settings
 */
void readSettings(CameraParameters& in_camParams,
                  TemplateGenerationSettings& in_tempGenSettings);
