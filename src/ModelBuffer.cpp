#include "ModelBuffer.h"

ModelBuffer::ModelBuffer(void* in_vertData, uint32_t in_numVertices,
                         void* in_indData, uint32_t in_numIndices,
                         uint8_t in_elementSize) {
  // 索引缓冲区Element Buffer Object(EBO)
  glGenBuffers(1, &indBufferId);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indBufferId);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, in_numIndices * in_elementSize,
               in_indData, GL_STATIC_DRAW);
  numIndices = in_numIndices;

  // 顶点(位置和颜色)缓冲区Vertex Array Object(VAO)
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vertBufferId);
  glBindBuffer(GL_ARRAY_BUFFER, vertBufferId);
  glBufferData(GL_ARRAY_BUFFER, in_numVertices * 2 * sizeof(glm::vec3),
               in_vertData, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3) * 2,
                        (void*)offsetof(glm::vec3, x));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3) * 2,
                        (void*)12);

  glBindVertexArray(0);
}

// 绑定VAO和EBO以供渲染
void ModelBuffer::bind() {
  glBindVertexArray(vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indBufferId);
}

// 解绑VAO和EBO
void ModelBuffer::unbind() {
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}
