#pragma once
#include <iostream>
#include <vector>

#include "CameraViewPoints.h"
#include "HighLevelLinemod.h"
#include "OpenglRender.h"
#include "defines.h"
#include "utility.h"

namespace hlm {
/**
 * @brief High level class covering the template generation
 *
 */
class TemplateGenerator {
 public:
  TemplateGenerator();
  ~TemplateGenerator();

  // Cleaning up function
  void cleanup();

  // Running the template generation and saving it to a file afterwards
  void run();

 private:
  OpenGLRender* opengl_;
  HighLevelLineMOD* line_;
  CameraViewPoints* camPoints_;
  std::vector<glm::vec3> camVertices_;

  std::vector<std::string> modelFiles_;
  std::string modelFolder_;
  uint16_t startDistance_;
  uint16_t endDistance_;
  uint16_t stepSize_;
  uint16_t subdivisions_;

  uint32_t numCameraVertices_;

  /**
   * @brief Create the viewpoints of the camera
   *
   * @param in_radiusToModel Distance to model in mm
   */
  void createCamViewPoints(float in_radiusToModel);

  /**
   * @brief Render the color and depth image
   *
   * @param[out] in_imgVec output image vector
   * @param in_modelIterator Index of model name
   * @param in_vertIterator Index of camera position from vector
   */
  void renderImages(std::vector<cv::Mat>& in_imgVec, uint16_t in_modelIterator,
                    uint16_t in_vertIterator);

  /**
   * @brief Utility function to print a progress bar
   *
   * @param in_percent Percent progress
   * @param in_mfile Name of model
   */
  void printProgBar(uint16_t in_percent, std::string const& in_mfile);

  // Calculate progress of model in percent
  uint16_t calculateCurrentPercent(uint16_t const& in_spehreRadius,
                                   uint16_t const& in_currentIteration);

  // Create a yaml settings file
  void writeSettings();
};
}  // namespace hlm