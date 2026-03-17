#pragma once
#define GLEW_STATIC
#include <GL/glew.h>

#include <cstddef>

#include "defines.h"

/**
 * @brief Struct to handel the model buffers for opengl
 *
 */
struct ModelBuffer {
  /**
   * @brief Construct a new Model Buffer object
   *
   */
  ModelBuffer(void* in_vertData, uint32_t in_numVertices, void* in_indData,
              uint32_t in_numIndices, uint8_t in_elementSize);

  /**
   * @brief Binding the model buffer for rendering
   *
   */
  void bind();

  /**
   * @brief Unbinding the buffer after rendering
   *
   */
  void unbind();

 public:
  GLuint numIndices;

 private:
  GLuint indBufferId;
  GLuint vertBufferId;
  GLuint vao;
};
