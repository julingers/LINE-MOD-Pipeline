#pragma once
#include <glm/glm.hpp>
#include <vector>

#include "defines.h"

const float goldenRatio = 1.61803398875;

/**
 * @brief Class to generate camera view points. Either from subdivided
 * icosaheadron or circle
 */
class CameraViewPoints {
  float radius_;
  float icosahedronPointA_;
  float icosahedronPointB_;
  uint8_t numSubdivisions_;

  std::vector<glm::vec3> vertices_;
  std::vector<Index> indices_;

 public:
  CameraViewPoints();
  ~CameraViewPoints();

  /**
   * @brief Create a Camera View Points
   *
   * @param in_radius Distance in millimeter between camera and object origin
   * @param in_subdivions Number of subdivisions for icosaeder
   */
  void createCameraViewPoints(float in_radius, uint8_t in_subdivions);

  /**
   * @brief Read symmetry properties of the object file
   *
   * @param in_modelFile Name of the model
   */
  void readModelProperties(std::string in_modelFile);

  // Get the Vertices
  std::vector<glm::vec3>& getVertices();

 private:
  ModelProperties modProps_;

  // Remove camera positions that generate an identical image for object
  void removeSuperfluousVertices();

  // Estimating icosahedron points depending on given radius
  void icosahedronPointsFromRadius();

  /**
   * @brief Create vertices for a rotation symmetrical object
   * @detail Degree distance between points is 60deg / number of subdivisons
   *
   */
  void createVerticesForRotSym();

  //  Create an Icosahedron
  void createIcosahedron(const float& icosahedronPointA,
                         const float& icosahedronPointB,
                         std::vector<glm::vec3>& vertices,
                         std::vector<Index>& indices);

  // Remove duplicate vertices after subdivison
  int32_t checkForDuplicate(uint32_t vertSize);

  // subdivide the created icosahedron
  void subdivide();

  // Adjust the distance of a subdivided vertex to match the radius
  void adjustVecToRadius(uint32_t index);
};
