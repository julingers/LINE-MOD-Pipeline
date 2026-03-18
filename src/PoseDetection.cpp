#include "PoseDetection.h"

PoseDetection::PoseDetection() {
  readSettings(camParams_, templateSettings_);

  filesInDirectory(modelFiles_, templateSettings_.modelFolder,
                   templateSettings_.modelFileEnding);
  opengl_ = new OpenGLRender(camParams_);
  line_ = new HighLevelLineMOD(camParams_, templateSettings_);
  icp_ = new HighLevelLinemodIcp(6, 0.1f, 2.5f, 8,
                                 templateSettings_.icpSubsamplingFactor,
                                 modelFiles_, templateSettings_.modelFolder);

  for (const auto& modelFile : modelFiles_) {
    opengl_->creatModBuffFromFiles(templateSettings_.modelFolder + modelFile);
  }
  readLinemodFromFile();
}

PoseDetection::~PoseDetection() { cleanup(); }

void PoseDetection::cleanup() {
  if (opengl_) {
    delete opengl_;
  }
  if (line_) {
    delete line_;
  }
  if (icp_) {
    delete icp_;
  }
  if (bench_) {
    delete bench_;
  }
}

void PoseDetection::detect(std::vector<cv::Mat>& in_imgs,
                           std::string const& in_className,
                           uint16_t const& in_numberOfObjects,
                           std::vector<ObjectPose>& in_objPose,
                           bool in_displayResults) {
  uint16_t numClassIndex = findIndexInVector(in_className, ids_);

  colorImg_ = in_imgs[0];
  depthImg_ = in_imgs[1];

  cv::Mat correctedTranslationColor = colorImg_.clone();
  cv::Mat correctedTranslationDepth = depthImg_.clone();
  translateImg(correctedTranslationColor,
               -camParams_.cx + camParams_.videoWidth / 2,
               -camParams_.cy + camParams_.videoHeight / 2);
  translateImg(correctedTranslationDepth,
               -camParams_.cx + camParams_.videoWidth / 2,
               -camParams_.cy + camParams_.videoHeight / 2);

  inputImg_.push_back(correctedTranslationColor);
  inputImg_.push_back(correctedTranslationDepth);

  finalObjectPoses_.clear();
  line_->detectTemplate(inputImg_, numClassIndex);
  detectedPoses_ = line_->getObjectPoses();
  if (!detectedPoses_.empty()) {
    for (auto& detectedPose : detectedPoses_) {
      if (templateSettings_.useIcp) {
        uint16_t bestPose = 0;
        icp_->prepareDepthForIcp(depthImg_, camParams_.cameraMatrix,
                                 detectedPose[0].boundingBox);
        icp_->registerToScene(detectedPose, numClassIndex);
        bool truePositivMatch =
            icp_->estimateBestMatch(correctedTranslationDepth, detectedPose,
                                    opengl_, numClassIndex, bestPose);
        if (truePositivMatch) {
          finalObjectPoses_.push_back(detectedPose[bestPose]);
        }
      } else {
        finalObjectPoses_.push_back(detectedPose[0]);
      }
      if (finalObjectPoses_.size() == in_numberOfObjects) {
        break;
      }
    }
    if (!finalObjectPoses_.empty()) {
      if (bench_) {
        float error = 1;
        error =
            bench_->calculateErrorHodan(correctedTranslationDepth, opengl_,
                                        finalObjectPoses_[0], numClassIndex);
        // error = bench_->calculateErrorLM(finalObjectPoses_[0]);
        // error = bench_->calculateErrorLMAmbigous(finalObjectPoses_[0]);
        std::cout << "Error: " << error << std::endl;
      }
      if (in_displayResults) {
        for (auto& finalObjectPose : finalObjectPoses_) {
          in_objPose.push_back(finalObjectPose);
          drawCoordinateSystem(colorImg_, camParams_.cameraMatrix, 75.0f,
                               finalObjectPose);
        }
      }
    }
  }
  if (in_displayResults) {
    if (bench_) {
      bench_->increaseImgCounter();
    }
    imshow("color", colorImg_);
    cv::waitKey(1);
  }

  inputImg_.clear();
}

void PoseDetection::setupBenchmark(std::string const& in_className) {
  bench_ = new Benchmark;
  uint16_t numClassIndex = findIndexInVector(in_className, ids_);
  bench_->loadModel(opengl_,
                    templateSettings_.modelFolder + modelFiles_[numClassIndex]);
}

uint16_t PoseDetection::findIndexInVector(
    std::string const& in_stringToFind,
    std::vector<std::string>& in_vectorToLookIn) {
  auto ind = std::find(in_vectorToLookIn.begin(), in_vectorToLookIn.end(),
                       in_stringToFind);
  return std::distance(in_vectorToLookIn.begin(), ind);
}

void PoseDetection::readLinemodFromFile() {
  line_->readLinemod();
  ids_ = line_->getClassIds();
  int num_classes = line_->getNumClasses();
  std::cout << "Loaded with " << num_classes << " classes and "
            << line_->getNumTemplates() << " templates\n"
            << std::endl;
  if (!ids_.empty()) {
    std::cout << "Class ids: " << std::endl;
    std::copy(ids_.begin(), ids_.end(),
              std::ostream_iterator<std::string>(std::cout, "\n"));
  }
  if (ids_ != modelFiles_) {
    std::cout
        << "ERROR::Models in file folder do not match with generated models!"
        << std::endl;
  }
}

void PoseDetection::drawCoordinateSystem(cv::Mat& in_srcDstImage,
                                         const cv::Mat& in_camMat,
                                         float in_coordinateSystemLength,
                                         ObjectPose& in_objPos) {
  cv::Mat rotMat;
  fromGLM2CV(toMat3(in_objPos.quaternions), &rotMat);

  cv::Vec3f tVec(in_objPos.translation.x, in_objPos.translation.y,
                 in_objPos.translation.z);
  cv::Mat rVec;
  std::vector<cv::Point3f> coordinatePoints;
  std::vector<cv::Point2f> projectedPoints;
  cv::Point3f center(0.0f, 0.0f, 0.0f);
  cv::Point3f xm(in_coordinateSystemLength, 0.0f, 0.0f);
  cv::Point3f ym(0.0f, in_coordinateSystemLength, 0.0f);
  cv::Point3f zm(0.0f, 0.0f, in_coordinateSystemLength);

  coordinatePoints.push_back(center);
  coordinatePoints.push_back(xm);
  coordinatePoints.push_back(ym);
  coordinatePoints.push_back(zm);

  cv::Rodrigues(rotMat, rVec);

  cv::projectPoints(coordinatePoints, rVec, tVec, in_camMat,
                    std::vector<double>(), projectedPoints);
  cv::line(in_srcDstImage, projectedPoints[0], projectedPoints[1],
           cv::Scalar(0, 0, 255), 2);
  cv::line(in_srcDstImage, projectedPoints[0], projectedPoints[2],
           cv::Scalar(0, 255, 0), 2);
  cv::line(in_srcDstImage, projectedPoints[0], projectedPoints[3],
           cv::Scalar(255, 0, 0), 2);
}

cv::Mat PoseDetection::translateImg(cv::Mat& in_img, int in_offsetx,
                                    int in_offsety) {
  cv::Mat trans_mat =
      (cv::Mat_<double>(2, 3) << 1, 0, in_offsetx, 0, 1, in_offsety);
  cv::warpAffine(in_img, in_img, trans_mat, in_img.size());
  return in_img;
}
