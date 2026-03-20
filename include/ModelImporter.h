#pragma once
// #include <vector>
#include <string>
// #include <cassert>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <iostream>

#include "defines.h"

/**
 * @brief Class to read the 3D-Models and place them in a Model struct
 *
 */
class ModelImporter {
 public:
  /**
   * @brief Construct a new Model Importer object
   *
   */
  ModelImporter();
  ~ModelImporter();

  /**
   * @brief Import, triangulate the model and place the vertecies, indecies and
   * colors in the Model struct
   *
   * @param in_modelFile
   * @param[out] in_model
   */
  void importModel(std::string const& in_modelFile, Model& in_model);

 private:
  Assimp::Importer* importer;

  void processNode(aiNode* in_node, const aiScene* in_scene, Model& in_model);
  void processMesh(aiMesh* mesh, const aiScene* scene, Model& in_model);
};
