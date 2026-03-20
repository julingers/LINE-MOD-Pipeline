#pragma once

#include <glog/logging.h>

#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <utility>

#include "defines.h"
#include "utility.h"

/**
 * @brief Class wraps template generation and detection. Builds on the opencv
 * linemod class
 */
class HighLevelLineMOD {
 public:
  /**
   * @brief Construct a new High Level LINE-MOD object
   *
   * @param in_camParams Contains the Camera Parameters
   * @param in_templateSettings Contains the settings for template generation
   * and detection
   */
  HighLevelLineMOD(CameraParameters const& in_camParams,
                   TemplateGenerationSettings const& in_templateSettings);
  ~HighLevelLineMOD();

  // Class Id getter
  std::vector<cv::String> getClassIds();

  // Number of classes getter
  uint16_t getNumClasses();

  // Number of Templates getter
  uint32_t getNumTemplates();

  /**
   * @brief Adding a template from input images
   * @return true Templates correctly extracted from image
   * @return false Returns false if the templates cant be created. Usually its
   * because they are too small
   */
  bool addTemplate(std::vector<cv::Mat>& in_images,
                   const std::string& in_modelName, glm::vec3 in_cameraPosition);

  /**
   * @brief Detect templates in the given images with the given class number
   * @return true
   * @return false could not find a template
   */
  bool detectTemplate(std::vector<cv::Mat>& in_imgs, uint16_t in_classNumber);

  /**
   * @brief Write the detecor and templates to a file called
   * "linemod_templates.yml.gz" and "linemod_tempPosFile.bin"
   */
  void writeLinemod();

  // Reading the templates and the detector from the written files
  void readLinemod();

  // Add the templates of the current class to a vector of templates
  void pushBackTemplates();

  // Reading the Object Poses after detection
  std::vector<std::vector<ObjectPose>> getObjectPoses();

 private:
  cv::Ptr<cv::linemod::Detector> detector_;

  bool onlyColorModality_;

  uint16_t videoWidth_;
  uint16_t videoHeight_;
  float cx_;
  float cy_;
  float fx_;
  float fy_;
  float fieldOfViewHeight_;

  std::vector<cv::Mat> inPlaneRotationMat_;
  int16_t lowerAngleStop_;
  int16_t upperAngleStop_;
  uint16_t angleStep_;
  uint16_t stepSize_;
  float detectorThreshold_;
  uint16_t percentToPassCheck_;
  uint16_t numberWantedPoses_;
  float radiusThresholdNewObject_;
  float discardGroupRatio_;
  bool useDepthImprovement_;
  float depthOffset_;

  glm::vec3 up_ = glm::vec3(0.0f, 1.0f, 0.0f);
  int32_t tempDepth_;

  struct Template {
    Template() {}

    Template(glm::vec3 tra, glm::quat qua, cv::Rect bb, uint16_t med)
        : translation(tra),
          quaternions(qua),
          boundingBox(std::move(bb)),
          medianDepth(med) {}

    glm::vec3 translation;
    glm::quat quaternions;
    cv::Rect boundingBox;
    uint16_t medianDepth;
  };

  struct PotentialMatch {
    PotentialMatch(cv::Point in_point, size_t in_indices)
        : position(std::move(in_point)) {
      matchIndices.push_back(in_indices);
    }

    cv::Point position;
    std::vector<uint32_t> matchIndices;
  };

  cv::Mat colorImgHue_;
  std::vector<Template> templates_;
  std::vector<std::vector<Template>> modelTemplates_;
  std::vector<std::vector<ObjectPose>> posesMultipleObj_;
  std::vector<cv::linemod::Match> matches_;
  std::vector<cv::linemod::Match> groupedMatches_;
  std::vector<PotentialMatch> potentialMatches_;
  std::vector<ModelProperties> modProps_;

  std::vector<std::string> modelFiles_;
  std::string modelFolder_;

  // Generation the matrices for image rotation
  void generateRotMatForInplaneRotation();

  /**
   * @brief Calculate the median or quartile of an opencv matrice
   *
   * @param in_mat The matrice to calculate
   * @param in_bb The bounding box describing the area to calculate the medain
   * @param in_medianPosition The wanted position of the sorted values. For
   * median this should be 2. For lower quartile it is 4.
   * @return uint16_t
   */
  uint16_t medianMat(cv::Mat const& in_mat, cv::Rect& in_bb,
                     uint8_t in_medianPosition);

  /**
   * @brief Calculation position and rotation from the camera position and the
   * in plane rotation angle
   *
   * @param[out] in_translation
   * @param[out] in_quats
   * @param in_cameraPosition
   * @param in_inplaneRot
   */
  void calculateTemplatePose(glm::vec3& in_translation, glm::quat& in_quats,
                             glm::vec3& in_cameraPosition,
                             int16_t& in_inplaneRot);

  /**
   * @brief The function converts the rotation from the opengl coordinate system
   * to the opencv one
   *
   * @param in_viewMat
   * @return glm::quat
   */
  glm::quat openglCoordinatesystem2opencv(glm::mat4& in_viewMat);

  /**
   * @brief Function that applies color and depth checks to the matches until a
   * set number of matches pass
   *
   * @param in_imgs
   * @param[out] in_objPoses
   * @return true True means that atleast one match passed the check
   * @return false False means that no match passed the checks
   */
  bool applyPostProcessing(std::vector<cv::Mat>& in_imgs,
                           std::vector<ObjectPose>& in_objPoses);

  /**
   * @brief Test if the color of a match is correct
   *
   * @param in_colImg Binary image. White pixel are the correct color
   * @param in_numMatch Index of the match in the matches vector
   * @param in_percentToPassCheck How many percent of the pixel have to be
   * correct
   * @return true
   * @return false
   */
  bool colorCheck(cv::Mat& in_hueImg, uint32_t& in_numMatch,
                  float in_percentCorrectColor);
  bool depthCheck(cv::Mat& in_depth, uint32_t& in_numMatch);

  /**
   * @brief Update the translation and rotation of a match depending on its 2D
   * position and place it in the pose vector
   *
   * @param in_numMatch Index of the match in the matches vector
   * @param[out] in_objPoses
   */
  void updateTranslationAndCreateObjectPose(
      uint32_t const& in_numMatch, std::vector<ObjectPose>& in_objPoses);

  /**
   * @brief Calculate the translation
   *
   * @param in_numMatch Index of the match in the matches vector
   * @param[out] in_position translation of the object
   * @param in_directDepth depth check adjusted distance between camera and
   * object
   */
  void calcPosition(uint32_t const& in_numMatch, glm::vec3& in_position,
                    float const& in_directDepth);

  /**
   * @brief Calculate the adjusted roatation in quaternions of the match
   *
   * @param in_numMatch Index of the match in the matches vector
   * @param in_position Updated translation vector
   * @param[out] in_quats Quaternions of the object
   */
  void calcRotation(uint32_t const& in_numMatch, glm::vec3 const& in_position,
                    glm::quat& in_quats);

  // Calculate the match origin position in pixel
  void matchToPixelCoord(uint32_t const& in_numMatch, float& in_x, float& in_y);

  // Calculate the pixel distance from origin to image center
  float pixelDistToCenter(float in_x, float in_y);

  /**
   * @brief Calculate the z-position of the object out of the direct distance
   * between object and camera
   */
  float calcTrueZ(float const& in_directDist, float const& in_angleFromCenter);

  /**
   * @brief Calculate a rough mask by estimating the convex hull of features
   *
   * @param in_match
   * @param[out] dst image with the mask
   */
  void templateMask(cv::linemod::Match const& in_match, cv::Mat& dst);

  // Sort matches with a similar position in the image into groups
  void groupSimilarMatches();

  /**
   * @brief Remove groups from the sorted list of matches with a low percentage
   * of elements
   */
  void discardSmallMatchGroups();

  /**
   * @brief Function to pick out elements from vector with a list of indices
   *
   * @param in_matches Vector to pick elements from
   * @param in_indices Vector of indices to pick from other vector
   * @return std::vector<cv::linemod::Match>
   */
  std::vector<cv::linemod::Match> elementsFromListOfIndices(
      std::vector<cv::linemod::Match>& in_matches,
      const std::vector<uint32_t>& in_indices);

  // Read the object color for the color check from corresponding file
  void readColorRanges();

  // Draws the features of a match on the image for debuging
  void drawResponse(const std::vector<cv::linemod::Template>& templates,
                    int num_modalities, cv::Mat& dst, const cv::Point& offset,
                    int T);
};
