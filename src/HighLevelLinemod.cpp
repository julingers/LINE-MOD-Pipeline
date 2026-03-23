#include "HighLevelLinemod.h"
namespace hlm {
HighLevelLineMOD::HighLevelLineMOD(
    CameraParameters const& in_camParams,
    TemplateGenerationSettings const& in_templateSettings)
    : onlyColorModality_(in_templateSettings.onlyUseColorModality),
      videoWidth_(in_camParams.videoWidth),
      videoHeight_(in_camParams.videoHeight),
      cx_(in_camParams.cx),
      cy_(in_camParams.cy),
      fx_(in_camParams.fx),
      fy_(in_camParams.fy),
      fieldOfViewHeight_(360.0f / CV_PI *
                         atanf(videoHeight_ / (2 * in_camParams.fy))),
      lowerAngleStop_(in_templateSettings.angleStart),
      upperAngleStop_(in_templateSettings.angleStop),
      angleStep_(in_templateSettings.angleStep),
      stepSize_(in_templateSettings.stepSize),
      modelFolder_(in_templateSettings.modelFolder),
      detectorThreshold_(in_templateSettings.detectorThreshold),
      percentToPassCheck_(in_templateSettings.percentToPassCheck),
      numberWantedPoses_(in_templateSettings.numberWantedPoses),
      radiusThresholdNewObject_(in_templateSettings.radiusThresholdNewObject),
      discardGroupRatio_(in_templateSettings.discardGroupRatio),
      useDepthImprovement_(in_templateSettings.useDepthImprovement),
      depthOffset_(in_templateSettings.depthOffset) {
  if (!onlyColorModality_) {
    std::vector<cv::Ptr<cv::linemod::Modality>> modality;
    // default Constructor Params: 10, 63, 55
    modality.emplace_back(cv::makePtr<cv::linemod::ColorGradient>());
    // default Constructor Params: 2000, 50, 63, 2
    modality.emplace_back(cv::makePtr<cv::linemod::DepthNormal>());

    // 金字塔采样步长，两层，第一层步长为5，第二层步长为8
    static const int T_DEFAULTS[] = {2, 8};
    detector_ = cv::makePtr<cv::linemod::Detector>(
        modality, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
  } else {
    std::vector<cv::Ptr<cv::linemod::Modality>> modality;
    modality.emplace_back(cv::makePtr<cv::linemod::ColorGradient>());
    // 金字塔采样步长，两层，第一层步长为2，第二层步长为8
    static const int T_DEFAULTS[] = {2, 8};
    detector_ = cv::makePtr<cv::linemod::Detector>(
        modality, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
  }

  generateRotMatForInplaneRotation();
}

HighLevelLineMOD::~HighLevelLineMOD() { detector_.release(); }

std::vector<cv::String> HighLevelLineMOD::getClassIds() {
  return detector_->classIds();
}

uint16_t HighLevelLineMOD::getNumClasses() { return detector_->numClasses(); }

uint32_t HighLevelLineMOD::getNumTemplates() {
  return detector_->numTemplates();
}

bool HighLevelLineMOD::addTemplate(std::vector<cv::Mat>& in_images,
                                   const std::string& in_modelName,
                                   glm::vec3 in_cameraPosition) {
  cv::Mat mask, maskRotated;
  std::vector<cv::Mat> templateImgs;
  cv::Mat colorRotated, depthRotated, colorToBinary;
  cv::threshold(in_images[0], colorToBinary, 1, 255, cv::THRESH_BINARY);
  cv::threshold(in_images[1], mask, 1, 65535, cv::THRESH_BINARY);
  mask.convertTo(mask, CV_8UC1);

  int cnt = 0;
  for (size_t q = 0; q < inPlaneRotationMat_.size(); q++) {
    cv::warpAffine(mask, maskRotated, inPlaneRotationMat_[q],
                   maskRotated.size(), cv::INTER_NEAREST);
    cv::warpAffine(colorToBinary, colorRotated, inPlaneRotationMat_[q],
                   colorToBinary.size());
    cv::warpAffine(in_images[1], depthRotated, inPlaneRotationMat_[q],
                   in_images[1].size(), cv::INTER_NEAREST);
    depthRotated.setTo(0, maskRotated == 0);

    // LOG(ERROR) << "images[0].type(): " << in_images[0].type()
    //            << ", colorToBinary.type(): " << colorToBinary.type()
    //            << ", colorRotated.type(): " << colorRotated.type()
    //            << ",in_images[1].type(): " << in_images[1].type()
    //            << ", depthRotated.type(): " << depthRotated.type()
    //            << ", maskRotated.type(): " << maskRotated.type();
    // double minv, maxv;
    // cv::minMaxLoc(depthRotated, &minv, &maxv);
    // LOG(ERROR) << "depth range: " << minv << " ~ " << maxv;
    // cv::imwrite("/mnt/hgfs/data/depthRotated.png", depthRotated);
    // cv::imwrite("/mnt/hgfs/data/maskRotated.png", maskRotated);

    templateImgs.push_back(colorRotated);
    if (!onlyColorModality_) {
      templateImgs.push_back(depthRotated);
    }
    // cv::erode(maskRotated, maskRotated, cv::Mat(), cv::Point(-1, -1), 1);
    cv::Rect boundingBox;
    uint64_t template_id = detector_->addTemplate(templateImgs, in_modelName,
                                                  maskRotated, &boundingBox);
    templateImgs.clear();

    if (template_id == -1) {
      std::cout << "ERROR::Cant create Template" << std::endl;
      return false;
    }

    // debug：模态特征可视化
    if (0) {
      const std::vector<cv::linemod::Template>& templates =
          detector_->getTemplates(in_modelName, template_id);
      LOG(INFO) << "template size: " << templates.size();
      for (size_t i = 0; i < templates.size(); i++) {
        LOG(INFO) << "feature " << i
                  << " size: " << templates[i].features.size();
        if (templates[i].features.empty()) {
          LOG(ERROR) << "template " << i << " has no features!";
        }
      }
      if (!onlyColorModality_) {
        cv::Mat colorCanvas;
        colorCanvas = colorRotated.clone();

        // cv::normalize(depthRotated.clone(), colorCanvas, 0, 255,
        //               cv::NORM_MINMAX);
        // colorCanvas.convertTo(colorCanvas, CV_8UC1);
        // cv::cvtColor(colorCanvas, colorCanvas, cv::COLOR_GRAY2BGR);
        for (const auto& f : templates[0].features) {
          // 特征点位置是相对于boundingbox的偏移
          cv::Point pt(f.x + boundingBox.x, f.y + boundingBox.y);
          cv::circle(colorCanvas, pt, 1, cv::Scalar(0, 255, 0), -1);
        }

        int label0_count = 0;
        for (const auto& f : templates[1].features) {
          if (f.label == 0) label0_count++;
          // 特征点位置是相对于boundingbox的偏移
          cv::Point pt(f.x + boundingBox.x, f.y + boundingBox.y);
          cv::circle(colorCanvas, pt, 1, cv::Scalar(0, 0, 255), -1);
        }
        LOG(INFO) << "depth features with label 0: " << label0_count;
        // cv::imshow("features", colorCanvas);
        // cv::waitKey(0);
      }
    }

    glm::vec3 translation;
    glm::quat quaternions;
    uint16_t medianDepth = medianMat(depthRotated, boundingBox, 5);
    // 计算当前平面的旋转角度（In-plane Rotation, 即相机围绕镜头光轴旋转）
    int16_t currentInplaneAngle = -(lowerAngleStop_ + q * angleStep_);
    // 计算模板在opencv坐标系下的6d位姿
    calculateTemplatePose(translation, quaternions, in_cameraPosition,
                          currentInplaneAngle);
    templates_.emplace_back(translation, quaternions, boundingBox, medianDepth);
  }
  return true;
}

void HighLevelLineMOD::templateMask(cv::linemod::Match const& in_match,
                                    cv::Mat& dst) {
  const std::vector<cv::linemod::Template>& templates =
      detector_->getTemplates(in_match.class_id, in_match.template_id);
  cv::Point offset(in_match.x, in_match.y);
  std::vector<cv::Point> points;
  uint16_t num_modalities = detector_->getModalities().size();
  for (int m = 0; m < num_modalities; ++m) {
    for (cv::linemod::Feature f : templates[m].features) {
      points.push_back(cv::Point(f.x, f.y) + offset);
    }
  }

  std::vector<cv::Point> hull;
  cv::convexHull(points, hull);

  dst = cv::Mat::zeros(cv::Size(videoWidth_, videoHeight_), CV_8U);
  const auto hull_count = (int)hull.size();
  const cv::Point* hull_pts = &hull[0];
  cv::fillPoly(dst, &hull_pts, &hull_count, 1, cv::Scalar(255));
}

static cv::Mat visualizeMatches(
    const cv::Mat& in_img, const std::vector<cv::linemod::Match>& matches,
    const cv::Ptr<cv::linemod::Detector>& detector) {
  cv::Mat img_show = in_img.clone();

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(0, 255);
  for (const auto& match : matches) {
    const auto templates =
        detector->getTemplates(match.class_id, match.template_id);
    if (templates.empty()) continue;

    const auto& t = templates[0];

    int minX = INT_MAX, minY = INT_MAX;
    int maxX = INT_MIN, maxY = INT_MIN;
    for (const auto& f : t.features) {
      minX = std::min(minX, (int)f.x);
      minY = std::min(minY, (int)f.y);
      maxX = std::max(maxX, (int)f.x);
      maxY = std::max(maxY, (int)f.y);
    }
    cv::Rect box(match.x + minX, match.y + minY, maxX - minX + 1,
                 maxY - minY + 1);

    box &= cv::Rect(0, 0, img_show.cols, img_show.rows);
    if (box.area() <= 0) continue;

    cv::Scalar color = cv::Scalar(dist(rng), dist(rng), dist(rng));
    cv::rectangle(img_show, box, color, 2);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << match.similarity;
    std::string label = oss.str();

    int baseline = 0;
    double fontScale = 0.5;
    int thickness = 1;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                        fontScale, thickness, &baseline);
    int textX = box.x;
    int textY = std::max(box.y - 5, textSize.height + 2);

    cv::rectangle(img_show, cv::Point(textX, textY - textSize.height - 2),
                  cv::Point(textX + textSize.width, textY + baseline), color,
                  cv::FILLED);
    cv::putText(img_show, label, cv::Point(textX, textY),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255),
                thickness);
  }

  return img_show;
}

static cv::Scalar hsv2bgr(float h, float s = 1.0f, float v = 1.0f) {
  // H: 0-180、 S,V: 0-255
  cv::Mat hsv(1, 1, CV_32FC3, cv::Scalar(h * 180, s * 255, v * 255));
  cv::Mat bgr;
  hsv.convertTo(hsv, CV_8UC3);
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(c[0], c[1], c[2]);  // BGR
}

static cv::Mat visualizeMatchGroups(
    const cv::Mat& in_img, const std::vector<cv::linemod::Match>& matches,
    const std::vector<PotentialMatch>& groups,
    const cv::Ptr<cv::linemod::Detector>& detector) {
  cv::Mat img_show = in_img.clone();
  size_t nGroups = groups.size();
  std::vector<cv::Scalar> groupColors;
  for (size_t i = 0; i < nGroups; ++i) {
    float hue = float(i) / nGroups;  // 均匀分布在0~1
    groupColors.push_back(hsv2bgr(hue));
  }

  for (size_t i = 0; i < nGroups; ++i) {
    const auto& group = groups[i];
    cv::Scalar color = groupColors[i];

    for (uint32_t idx : group.matchIndices) {
      const auto& match = matches[idx];
      const auto templates =
          detector->getTemplates(match.class_id, match.template_id);
      if (templates.empty()) continue;
      const auto& t = templates[0];

      int minX = INT_MAX, minY = INT_MAX;
      int maxX = INT_MIN, maxY = INT_MIN;
      for (const auto& f : t.features) {
        minX = std::min(minX, (int)f.x);
        minY = std::min(minY, (int)f.y);
        maxX = std::max(maxX, (int)f.x);
        maxY = std::max(maxY, (int)f.y);
      }
      cv::Rect box(match.x + minX, match.y + minY, maxX - minX + 1,
                   maxY - minY + 1);
      box &= cv::Rect(0, 0, img_show.cols, img_show.rows);
      cv::rectangle(img_show, box, color, 2);
      cv::circle(img_show, group.position, 3, color, cv::FILLED);
    }
  }
  return img_show;
}

bool HighLevelLineMOD::detectTemplate(std::vector<cv::Mat>& in_imgs,
                                      uint16_t in_classNumber,
                                      bool enable_visualization) {
  templates_ = modelTemplates_[in_classNumber];
  posesMultipleObj_.clear();
  cv::Mat tmpDepth;
  bool depthCheckForColorDetector = false;

  const std::vector<std::string> currentClass(
      1, detector_->classIds()[in_classNumber]);
  if (onlyColorModality_ && in_imgs.size() == 2) {
    LOG(INFO) << "Only using color modality.";
    tmpDepth = in_imgs[1];
    in_imgs.pop_back();
    depthCheckForColorDetector = true;
  } else {
    LOG(INFO) << "Using color and depth modality.";
  }
  detector_->match(in_imgs, detectorThreshold_, matches_, currentClass);
  if (depthCheckForColorDetector) {
    in_imgs.push_back(tmpDepth);
  }

  // 处理匹配结果，类似NMS思想的分组和筛选，最后进行后处理得到物体位姿
  LOG(WARNING) << "matches_ size: " << matches_.size();

  if (!matches_.empty()) {
    if (enable_visualization) {
      auto img_show = visualizeMatches(in_imgs[0], matches_, detector_);
      cv::imshow("initial matches visualization", img_show);
      cv::waitKey(1);
    }

    cv::cvtColor(in_imgs[0], colorImgHue_, cv::COLOR_BGR2HSV);
    cv::inRange(colorImgHue_, modProps_[in_classNumber].lowerColorRange,
                modProps_[in_classNumber].upperColorRange, colorImgHue_);

    // 聚类分组
    groupSimilarMatches();
    if (enable_visualization) {
      auto img_show = visualizeMatchGroups(in_imgs[0], matches_,
                                           potentialMatches_, detector_);
      cv::imshow("after groupSimilarMatches visualization", img_show);
      cv::waitKey(1);
    }
    // 过滤小簇
    discardSmallMatchGroups();
    if (enable_visualization) {
      auto img_show = visualizeMatchGroups(in_imgs[0], matches_,
                                           potentialMatches_, detector_);
      cv::imshow("after discardSmallMatchGroups visualization", img_show);
      cv::waitKey(0);
    }

    LOG(WARNING) << "potentialMatches_ size: " << potentialMatches_.size();
    for (auto& potentialMatch : potentialMatches_) {
      groupedMatches_ =
          elementsFromListOfIndices(matches_, potentialMatch.matchIndices);

      // 后处理
      std::vector<ObjectPose> objPoses;
      applyPostProcessing(in_imgs, objPoses);
      if (!objPoses.empty()) {
        posesMultipleObj_.push_back(objPoses);
      }
    }

    LOG(WARNING) << "posesMultipleObj_ size: " << posesMultipleObj_.size();
    // DRAW FEATURES OF BEST LINEMOD MATCH
    if (!posesMultipleObj_.empty()) {
      for (auto& potentialMatch : potentialMatches_) {
        const std::vector<cv::linemod::Template>& templates =
            detector_->getTemplates(
                matches_[potentialMatch.matchIndices[0]].class_id,
                matches_[potentialMatch.matchIndices[0]].template_id);
        drawResponse(templates, 1, in_imgs[0], potentialMatch.position,
                     detector_->getT(0));
      }
    }
    return true;
  }
  return false;
}

std::vector<cv::linemod::Match> HighLevelLineMOD::elementsFromListOfIndices(
    std::vector<cv::linemod::Match>& in_matches,
    const std::vector<uint32_t>& in_indices) {
  std::vector<cv::linemod::Match> tmpMatches;
  tmpMatches.reserve(in_indices.size());
  for (unsigned int in_indice : in_indices) {
    tmpMatches.push_back(in_matches[in_indice]);
  }
  return tmpMatches;
}

// 基于空间距离做聚类
void HighLevelLineMOD::groupSimilarMatches() {
  potentialMatches_.clear();
  for (size_t i = 0; i < matches_.size(); i++) {
    cv::Point matchPosition = cv::Point(matches_[i].x, matches_[i].y);
    uint16_t numCurrentGroups = potentialMatches_.size();
    bool foundGroupForMatch = false;

    for (size_t q = 0; q < numCurrentGroups; q++) {
      if (cv::norm(matchPosition - potentialMatches_[q].position) <
          radiusThresholdNewObject_) {
        potentialMatches_[q].matchIndices.push_back(i);
        foundGroupForMatch = true;
        break;
      }
    }
    if (!foundGroupForMatch) {
      potentialMatches_.emplace_back(cv::Point(matches_[i].x, matches_[i].y),
                                     i);
    }
  }
  LOG(WARNING) << "potentialMatches_ size after groupSimilarMatches: "
               << potentialMatches_.size();
}

// 按规模过滤小簇
void HighLevelLineMOD::discardSmallMatchGroups() {
  uint32_t numMatchGroups = potentialMatches_.size();
  uint32_t biggestGroup = 0;
  std::vector<PotentialMatch> tmp;
  for (size_t i = 0; i < numMatchGroups; i++) {
    if (potentialMatches_[i].matchIndices.size() > biggestGroup) {
      biggestGroup = potentialMatches_[i].matchIndices.size();
    }
  }
  for (size_t j = 0; j < numMatchGroups; j++) {
    float ratioToBiggestGroup =
        potentialMatches_[j].matchIndices.size() * 100 / biggestGroup;
    if (ratioToBiggestGroup > discardGroupRatio_) {
      tmp.push_back(potentialMatches_[j]);
    }
  }
  potentialMatches_.swap(tmp);
}

void HighLevelLineMOD::writeLinemod() {
  std::string posfix = onlyColorModality_ ? "color" : "color_depth";

  std::string filename = "linemod_templates_" + posfix + ".yml.gz";
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector_->write(fs);

  std::vector<cv::String> ids = detector_->classIds();
  fs << "classes" << "[";
  for (const auto& id : ids) {
    fs << "{";
    detector_->writeClass(id, fs);
    fs << "}";
  }
  fs << "]";

  std::ofstream templatePositionFile("linemod_tempPosFile_" + posfix + ".bin",
                                     std::ios::binary | std::ios::out);
  uint32_t numTempVecs = modelTemplates_.size();
  templatePositionFile.write((char*)&numTempVecs, sizeof(uint32_t));
  for (const auto& modelTemplate : modelTemplates_) {
    uint64_t numTemp = modelTemplate.size();
    templatePositionFile.write((char*)&numTemp, sizeof(uint64_t));
    for (uint64_t i = 0; i < numTemp; i++) {
      templatePositionFile.write((char*)&modelTemplate[i], sizeof(Template));
    }
  }
  templatePositionFile.close();
}

void HighLevelLineMOD::readLinemod() {
  std::string posfix = onlyColorModality_ ? "color" : "color_depth";

  templates_.clear();
  modelTemplates_.clear();
  std::string filename = "linemod_templates_" + posfix + ".yml.gz";
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector_->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (auto&& i : fn) {
    detector_->readClass(i);
  }

  std::ifstream input = std::ifstream("linemod_tempPosFile_" + posfix + ".bin",
                                      std::ios::in | std::ios::binary);
  uint32_t numTempVecs;
  input.read((char*)&numTempVecs, sizeof(uint32_t));
  for (uint32_t numTempVec = 0; numTempVec < numTempVecs; numTempVec++) {
    uint64_t numTemp;
    input.read((char*)&numTemp, sizeof(uint64_t));
    Template tp;
    for (uint64_t i = 0; i < numTemp; i++) {
      input.read((char*)&tp, sizeof(Template));
      templates_.push_back(tp);
    }
    modelTemplates_.push_back(templates_);
    templates_.clear();
  }
  input.close();
  readColorRanges();
}

std::vector<std::vector<ObjectPose>> HighLevelLineMOD::getObjectPoses() {
  return posesMultipleObj_;
}

void HighLevelLineMOD::generateRotMatForInplaneRotation() {
  for (int16_t angle = lowerAngleStop_; angle <= upperAngleStop_;
       angle = angle + angleStep_) {
    cv::Point2f srcCenter(videoWidth_ / 2, videoHeight_ / 2);
    inPlaneRotationMat_.push_back(getRotationMatrix2D(srcCenter, angle, 1.0f));
  }
}

uint16_t HighLevelLineMOD::medianMat(cv::Mat const& in_mat, cv::Rect& in_bb,
                                     uint8_t in_medianPosition) {
  // 处理深度值为0的像素
  cv::Mat invBinRot;  // Turn depth values 0 into 65535
  cv::threshold(in_mat, invBinRot, 1, 65535,
                cv::THRESH_BINARY);  // pixel >=1 becomes 65535, 0 stays 0
  invBinRot = 65535 - invBinRot;     // 65535 becomes 0, 0 stays 65535
  cv::Mat croppedDepth =
      in_mat + invBinRot;  // pixel >=1 stays the same, 0 becomes 65535
  croppedDepth = croppedDepth(in_bb);

  std::vector<uint16_t> vecFromMat(croppedDepth.begin<uint16_t>(),
                                   croppedDepth.end<uint16_t>());

  if (vecFromMat.empty() || in_medianPosition == 0) {
    std::cout << "ERROR: No pixels in bounding box or median position is 0"
              << std::endl;
    return 0;
  }

  // 提取第targetIndex分位数的值
  size_t targetIndex = vecFromMat.size() / in_medianPosition;
  std::nth_element(vecFromMat.begin(), vecFromMat.begin() + targetIndex,
                   vecFromMat.end());
  return vecFromMat[targetIndex];
}

void HighLevelLineMOD::calculateTemplatePose(glm::vec3& in_translation,
                                             glm::quat& in_quats,
                                             glm::vec3& in_cameraPosition,
                                             int16_t& in_inplaneRot) {
  // 物体始终位于原点，相机绕物体旋转，因此平移向量的x和y分量为0，z分量为相机位置与原点的距离
  in_translation.x = 0.0f;
  in_translation.y = 0.0f;
  in_translation.z = glm::length(in_cameraPosition);

  // 防奇异点，万向锁
  if (in_cameraPosition[0] == 0 && in_cameraPosition[2] == 0) {
    // Looking straight up or down fails the cross product
    in_cameraPosition[0] = 0.00000000001;
  }
  glm::vec3 camUp = glm::normalize(
      glm::cross(in_cameraPosition, glm::cross(in_cameraPosition, up_)));
  glm::vec3 rotatedUp = glm::rotate(-camUp, glm::radians((float)in_inplaneRot),
                                    glm::normalize(in_cameraPosition));
  glm::mat4 view = glm::lookAt(in_cameraPosition, glm::vec3(0.0f), rotatedUp);
  in_quats = openglCoordinatesystem2opencv(view);
}

glm::quat HighLevelLineMOD::openglCoordinatesystem2opencv(
    glm::mat4& in_viewMat) {
  glm::mat4 coordinateTransform(1.0f);
  coordinateTransform[1][1] = -1.0f;
  coordinateTransform[2][2] = -1.0f;
  // glm::quat tempQuat = glm::toQuat(coordinateTransform * in_viewMat);
  glm::quat tempQuat = glm::toQuat(
      glm::transpose(glm::transpose(in_viewMat) * coordinateTransform));
  return tempQuat;
}

bool HighLevelLineMOD::applyPostProcessing(
    std::vector<cv::Mat>& in_imgs, std::vector<ObjectPose>& in_objPoses) {
  for (uint32_t i = 0; i < groupedMatches_.size(); i++) {
    if (!onlyColorModality_) {
      if (colorCheck(colorImgHue_, i, percentToPassCheck_) &&
          depthCheck(in_imgs[1], i)) {
        updateTranslationAndCreateObjectPose(i, in_objPoses);
      }
    } else if (onlyColorModality_ && in_imgs.size() == 2) {
      if (colorCheck(colorImgHue_, i, percentToPassCheck_) &&
          depthCheck(in_imgs[1], i)) {
        updateTranslationAndCreateObjectPose(i, in_objPoses);
      }
    } else {
      if (colorCheck(colorImgHue_, i, percentToPassCheck_)) {
        updateTranslationAndCreateObjectPose(i, in_objPoses);
      }
    }
    if (in_objPoses.size() == numberWantedPoses_) {
      return true;
    }
    if (i == (groupedMatches_.size() - 1)) {
      if (!in_objPoses.empty()) {
        return true;
      }
    }
  }
  return false;
}

bool HighLevelLineMOD::colorCheck(cv::Mat& in_colImg, uint32_t& in_numMatch,
                                  float in_percentToPassCheck) {
  cv::Mat croppedImage;
  cv::Mat mask;
  templateMask(groupedMatches_[in_numMatch], mask);
  bitwise_and(in_colImg, mask, croppedImage);

  float nonZer = countNonZero(croppedImage) * 100 / countNonZero(mask);
  return nonZer > in_percentToPassCheck;
}

bool HighLevelLineMOD::depthCheck(cv::Mat& in_depth, uint32_t& in_numMatch) {
  if (useDepthImprovement_) {
    cv::Rect bb(
        groupedMatches_[in_numMatch].x, groupedMatches_[in_numMatch].y,
        templates_[groupedMatches_[in_numMatch].template_id].boundingBox.width,
        templates_[groupedMatches_[in_numMatch].template_id]
            .boundingBox.height);
    int32_t depthDiff =
        (int32_t)medianMat(in_depth, bb, 5) -
        (int32_t)templates_[groupedMatches_[in_numMatch].template_id]
            .medianDepth -
        depthOffset_;
    tempDepth_ =
        templates_[groupedMatches_[in_numMatch].template_id].translation.z +
        depthDiff;
    return abs(depthDiff) < stepSize_;
  } else {
    tempDepth_ =
        templates_[groupedMatches_[in_numMatch].template_id].translation.z;
    return true;
  }
}

void HighLevelLineMOD::updateTranslationAndCreateObjectPose(
    uint32_t const& in_numMatch, std::vector<ObjectPose>& in_objPoses) {
  glm::vec3 updatedTanslation;
  glm::quat updatedRotation;
  calcPosition(in_numMatch, updatedTanslation, tempDepth_);
  calcRotation(in_numMatch, updatedTanslation, updatedRotation);
  cv::Rect boundingBox(
      groupedMatches_[in_numMatch].x, groupedMatches_[in_numMatch].y,
      templates_[groupedMatches_[in_numMatch].template_id].boundingBox.width,
      templates_[groupedMatches_[in_numMatch].template_id].boundingBox.height);
  in_objPoses.emplace_back(updatedTanslation, updatedRotation, boundingBox);
}

void HighLevelLineMOD::calcPosition(uint32_t const& in_numMatch,
                                    glm::vec3& in_position,
                                    float const& in_directDepth) {
  float pixelX, pixelY;
  matchToPixelCoord(in_numMatch, pixelX, pixelY);
  float offsetFromCenter = pixelDistToCenter(pixelX, pixelY);
  in_position.z = calcTrueZ(in_directDepth, offsetFromCenter);

  float mmOffsetFromCenter = in_position.z / fy_;
  in_position.x = (pixelX - videoWidth_ / 2) * mmOffsetFromCenter;
  in_position.y = (pixelY - videoHeight_ / 2) * mmOffsetFromCenter;
}

void HighLevelLineMOD::calcRotation(uint32_t const& in_numMatch,
                                    glm::vec3 const& in_position,
                                    glm::quat& in_quats) {
  glm::mat4 adjustRotation =
      lookAt(glm::vec3(-in_position.x, -in_position.y, in_position.z),
             glm::vec3(0.f), up_);
  in_quats = toQuat(
      adjustRotation *
      toMat4(templates_[groupedMatches_[in_numMatch].template_id].quaternions));
}

void HighLevelLineMOD::matchToPixelCoord(uint32_t const& in_numMatch,
                                         float& in_x, float& in_y) {
  in_x = (groupedMatches_[in_numMatch].x + videoWidth_ / 2 -
          templates_[groupedMatches_[in_numMatch].template_id].boundingBox.x);
  in_y = (groupedMatches_[in_numMatch].y + videoHeight_ / 2 -
          templates_[groupedMatches_[in_numMatch].template_id].boundingBox.y);
}

float HighLevelLineMOD::pixelDistToCenter(float in_x, float in_y) {
  in_x -= videoWidth_ / 2;
  in_y -= videoHeight_ / 2;
  return sqrt(in_x * in_x + in_y * in_y);
}

float HighLevelLineMOD::calcTrueZ(float const& in_directDist,
                                  float const& in_distFromCenter) {
  return sqrt(in_directDist * in_directDist -
              (in_distFromCenter * in_distFromCenter));
}

void HighLevelLineMOD::pushBackTemplates() {
  // 将当前模型的模板添加到modelTemplates_中，并清空templates_以准备下个模型的模板生成
  modelTemplates_.push_back(templates_);
  templates_.clear();
}

void HighLevelLineMOD::readColorRanges() {
  modProps_.clear();
  modelFiles_ = detector_->classIds();
  for (uint16_t i = 0; i < detector_->numClasses(); i++) {
    ModelProperties tmpModProp;
    std::string filename = modelFolder_ +
                           modelFiles_[i].substr(0, modelFiles_[i].size() - 4) +
                           ".yml";
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    cv::Vec3b tempVec;

    fs["lower color range"] >> tmpModProp.lowerColorRange;
    fs["upper color range"] >> tmpModProp.upperColorRange;
    fs["has rotational symmetry"] >> tmpModProp.rotationallySymmetrical;
    fs["planes of symmetry"] >> tempVec;
    tmpModProp.planesOfSymmetry = glm::vec3(tempVec[0], tempVec[1], tempVec[2]);

    modProps_.emplace_back(tmpModProp);
  }
}

void HighLevelLineMOD::drawResponse(
    const std::vector<cv::linemod::Template>& templates, int num_modalities,
    cv::Mat& dst, const cv::Point& offset, int T) {
  static const cv::Scalar COLORS[5] = {CV_RGB(0, 0, 255), CV_RGB(0, 255, 0),
                                       CV_RGB(255, 255, 0), CV_RGB(255, 140, 0),
                                       CV_RGB(0, 0, 255)};

  for (int m = 0; m < num_modalities; ++m) {
    cv::Scalar color = COLORS[m];
    for (const auto f : templates[m].features) {
      const cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
    }
  }
}
}  // namespace hlm
