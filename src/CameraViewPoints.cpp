#include "CameraViewPoints.h"

CameraViewPoints::CameraViewPoints() {}

CameraViewPoints::~CameraViewPoints() {}

void CameraViewPoints::createCameraViewPoints(float in_radius,
                                              uint8_t in_subdivions) {
  vertices_.clear();
  indices_.clear();
  radius_ = in_radius;
  numSubdivisions_ = in_subdivions;  // 正二十面体的细分数量

  if (modProps.rotationallySymmetrical) {
    // 旋转对称体，圆周采样
    createVerticesForRotSym();
  } else {
    // 非旋转对称体，球面均匀采样（正二十面体细分）
    icosahedronPointsFromRadius();
    createIcosahedron(icosahedronPointA_, icosahedronPointB_, vertices_,
                      indices_);  // 构造正二十面体
    vertices_.reserve((20 * pow(4, (uint32_t)numSubdivisions_)) / 2 + 2);
    indices_.reserve(20 * pow(4, (uint32_t)numSubdivisions_));
    subdivide();  // 三角面细分逼近完美球体
  }
  removeSuperfluousVertices();  // 移除多余顶点
}

void CameraViewPoints::removeSuperfluousVertices() {
  std::vector<glm::vec3> tmpVertices;
  for (const auto& vertice : vertices_) {
    glm::vec3 tmpVertice = vertice * modProps.planesOfSymmetry;
    bool allElementsPositive = true;
    if (tmpVertice.x < 0 || tmpVertice.y < 0 || tmpVertice.z < 0) {
      allElementsPositive = false;
    }
    if (allElementsPositive) {
      tmpVertices.push_back(vertice);
    }
  }
  vertices_.clear();
  vertices_ = tmpVertices;
}

std::vector<glm::vec3>& CameraViewPoints::getVertices() { return vertices_; }

void CameraViewPoints::readModelProperties(std::string in_modelFile) {
  std::string filename =
      in_modelFile.substr(0, in_modelFile.size() - 4) + ".yml";
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  cv::Vec3b tempVec;

  fs["lower color range"] >> modProps.lowerColorRange;
  fs["upper color range"] >> modProps.upperColorRange;
  fs["has rotational symmetry"] >> modProps.rotationallySymmetrical;
  fs["planes of symmetry"] >> tempVec;
  modProps.planesOfSymmetry = glm::vec3(tempVec[0], tempVec[1], tempVec[2]);
}

void CameraViewPoints::icosahedronPointsFromRadius() {
  icosahedronPointA_ =
      sqrt((radius_ * radius_) / (goldenRatio * goldenRatio + 1));
  icosahedronPointB_ = icosahedronPointA_ * goldenRatio;
}

void CameraViewPoints::createVerticesForRotSym() {
  for (uint16_t i = 0; i < 360; i = i + (60 / pow(2, numSubdivisions_))) {
    vertices_.emplace_back(0.0f, sin(i * CV_PI / 180.0f) * radius_,
                           cos(i * CV_PI / 180.0f) * radius_);
  }
}

void CameraViewPoints::createIcosahedron(const float& icosahedronPointA,
                                         const float& icosahedronPointB,
                                         std::vector<glm::vec3>& vertices,
                                         std::vector<Index>& indices) {
  vertices.emplace_back(-icosahedronPointA, 0.0f, icosahedronPointB);
  vertices.emplace_back(icosahedronPointA, 0.0f, icosahedronPointB);
  vertices.emplace_back(-icosahedronPointA, 0.0f, -icosahedronPointB);
  vertices.emplace_back(icosahedronPointA, 0.0f, -icosahedronPointB);

  vertices.emplace_back(0.0f, icosahedronPointB, icosahedronPointA);
  vertices.emplace_back(0.0f, icosahedronPointB, -icosahedronPointA);
  vertices.emplace_back(0.0f, -icosahedronPointB, icosahedronPointA);
  vertices.emplace_back(0.0f, -icosahedronPointB, -icosahedronPointA);

  vertices.emplace_back(icosahedronPointB, icosahedronPointA, 0.0f);
  vertices.emplace_back(-icosahedronPointB, icosahedronPointA, 0.0f);
  vertices.emplace_back(icosahedronPointB, -icosahedronPointA, 0.0f);
  vertices.emplace_back(-icosahedronPointB, -icosahedronPointA, 0.0f);

  indices.push_back(Index{0, 4, 1});
  indices.push_back(Index{0, 9, 4});
  indices.push_back(Index{9, 5, 4});
  indices.push_back(Index{4, 5, 8});
  indices.push_back(Index{4, 8, 1});

  indices.push_back(Index{8, 10, 1});
  indices.push_back(Index{8, 3, 10});
  indices.push_back(Index{5, 3, 8});
  indices.push_back(Index{5, 2, 3});
  indices.push_back(Index{2, 7, 3});

  indices.push_back(Index{7, 10, 3});
  indices.push_back(Index{7, 6, 10});
  indices.push_back(Index{7, 11, 6});
  indices.push_back(Index{11, 0, 6});
  indices.push_back(Index{0, 1, 6});

  indices.push_back(Index{6, 1, 10});
  indices.push_back(Index{9, 0, 11});
  indices.push_back(Index{9, 11, 2});
  indices.push_back(Index{9, 2, 5});
  indices.push_back(Index{7, 2, 11});
}

int32_t CameraViewPoints::checkForDuplicate(uint32_t vertSize) {
  uint32_t index = -1;

#pragma omp parallel for
  for (int32_t i = 0; i < vertSize; i++) {
    if (abs(vertices_[vertSize].x - vertices_[i].x) < 1e-6 &&
        abs(vertices_[vertSize].y == vertices_[i].y) < 1e-6 &&
        abs(vertices_[vertSize].z == vertices_[i].z) < 1e-6) {
      index = i;
    }
  }
  return index;
}

void CameraViewPoints::subdivide() {
  // 对20个三角面进行numSubdivisions次细分，把原来的1个大三角形替换成4个小三角形
  for (uint8_t j = 0; j < numSubdivisions_; j++) {
    uint32_t numFaces = indices_.size();
    uint32_t currentVertSize = vertices_.size();
    for (uint32_t i = 0; i < numFaces; i++) {
      // 对于每一个三角形面(顶点为A,B,C)，找到中点(AB中点,BC中点,AC中点)
      vertices_.emplace_back(
          (vertices_[indices_[i].a].x + vertices_[indices_[i].b].x) / 2,
          (vertices_[indices_[i].a].y + vertices_[indices_[i].b].y) / 2,
          (vertices_[indices_[i].a].z + vertices_[indices_[i].b].z) / 2);
      adjustVecToRadius(currentVertSize);  // 球面投影

      // 相邻的两个三角形会计算出同一条边上的同一个中点。使用OpenMP并行计算遍历检查点是否已经存在，避免存储两个相同的点
      uint32_t duplicateIndex = checkForDuplicate(currentVertSize);
      uint32_t abIndex = 0;
      if (duplicateIndex != -1) {
        vertices_.pop_back();
        abIndex = duplicateIndex;
      } else {
        abIndex = currentVertSize;
        currentVertSize++;
      }

      vertices_.emplace_back(
          (vertices_[indices_[i].c].x + vertices_[indices_[i].b].x) / 2,
          (vertices_[indices_[i].c].y + vertices_[indices_[i].b].y) / 2,
          (vertices_[indices_[i].c].z + vertices_[indices_[i].b].z) / 2);
      adjustVecToRadius(currentVertSize);

      uint32_t bcIndex = 0;
      duplicateIndex = checkForDuplicate(currentVertSize);
      if (duplicateIndex != -1) {
        vertices_.pop_back();
        bcIndex = duplicateIndex;
      } else {
        bcIndex = currentVertSize;
        currentVertSize++;
      }

      vertices_.emplace_back(
          (vertices_[indices_[i].a].x + vertices_[indices_[i].c].x) / 2,
          (vertices_[indices_[i].a].y + vertices_[indices_[i].c].y) / 2,
          (vertices_[indices_[i].a].z + vertices_[indices_[i].c].z) / 2);
      adjustVecToRadius(currentVertSize);

      uint32_t acIndex = 0;
      duplicateIndex = checkForDuplicate(currentVertSize);
      if (duplicateIndex != -1) {
        vertices_.pop_back();
        acIndex = duplicateIndex;
      } else {
        acIndex = currentVertSize;
        currentVertSize++;
      }

      indices_.push_back(Index{indices_[i].a, abIndex, acIndex});
      indices_.push_back(Index{indices_[i].b, abIndex, bcIndex});
      indices_.push_back(Index{indices_[i].c, bcIndex, acIndex});
      indices_[i] = {abIndex, bcIndex, acIndex};
    }
  }
}

void CameraViewPoints::adjustVecToRadius(uint32_t index) {
  // 计算点到圆心的距离，将其归一化（除以自身长度)并乘以radius
  float adjustValue = sqrt(vertices_[index].x * vertices_[index].x +
                           vertices_[index].y * vertices_[index].y +
                           vertices_[index].z * vertices_[index].z) /
                      radius_;
  vertices_[index].x /= adjustValue;
  vertices_[index].y /= adjustValue;
  vertices_[index].z /= adjustValue;
}
