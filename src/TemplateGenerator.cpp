#include "TemplateGenerator.h"

#include <opencv2/imgcodecs.hpp>

TemplateGenerator::TemplateGenerator() {
  CameraParameters camParams;
  TemplateGenerationSettings templateSettings;
  readSettings(camParams, templateSettings);
  modelFolder_ = templateSettings.modelFolder;
  startDistance_ = templateSettings.startDistance;
  endDistance_ = templateSettings.endDistance;
  stepSize_ = templateSettings.stepSize;
  subdivisions_ = templateSettings.subdivisions;

  opengl_ = new OpenGLRender(camParams);
  line_ = new HighLevelLineMOD(camParams, templateSettings);
  camPoints_ = new CameraViewPoints();
  filesInDirectory(modelFiles_, modelFolder_, templateSettings.modelFileEnding);
}

TemplateGenerator::~TemplateGenerator() { cleanup(); }

void TemplateGenerator::cleanup() {
  if (opengl_) {
    delete opengl_;
  }
  if (line_) {
    delete line_;
  }
  if (camPoints_) {
    delete camPoints_;
  }
}

void TemplateGenerator::run() {
  int cnt = 0;
  // 对于每个模型文件
  for (size_t i = 0; i < modelFiles_.size(); i++) {
    camPoints_->readModelProperties(modelFolder_ + modelFiles_[i]);
    opengl_->creatModBuffFromFiles(modelFolder_ + modelFiles_[i]);

    // 从startDistance_到endDistance_，以stepSize_为步长的方式，生成相机位置
    for (uint16_t radiusToModel = startDistance_; radiusToModel <= endDistance_;
         radiusToModel += stepSize_) {
      // 获得采样后的相机位置camVertices_，以及相机位置的数量numCameraVertices_
      createCamViewPoints(radiusToModel);

      // 对于每个相机位置，渲染出对应的color和depth图像，并将它们添加到line_中
      for (size_t j = 0; j < numCameraVertices_; j++) {
        printProgBar(calculateCurrentPercent(radiusToModel, j), modelFiles_[i]);

        std::vector<cv::Mat> images;
        renderImages(images, i, j);
        line_->addTemplate(images, modelFiles_[i], camVertices_[j], &cnt);
        // line_->addTemplate(images, modelFiles_[i], camVertices_[j]);
        cnt++;
      }
    }
    line_->pushBackTemplates();
  }
  std::cout << "renderImages cnt(color same as depth): " << cnt << std::endl;
  line_->writeLinemod();
}

void TemplateGenerator::createCamViewPoints(float in_radiusToModel) {
  camPoints_->createCameraViewPoints(in_radiusToModel, subdivisions_);
  camVertices_ = camPoints_->getVertices();
  numCameraVertices_ = camVertices_.size();
}

void TemplateGenerator::renderImages(std::vector<cv::Mat>& in_imgVec,
                                     uint16_t in_modelIterator,
                                     uint16_t in_vertIterator) {
  in_imgVec.clear();
  opengl_->renderDepthToFrontBuff(in_modelIterator,
                                  glm::vec3(camVertices_[in_vertIterator]));
  cv::Mat depth = opengl_->getDepthImgFromBuff();
  opengl_->renderColorToFrontBuff(in_modelIterator,
                                  glm::vec3(camVertices_[in_vertIterator]));
  cv::Mat color = opengl_->getColorImgFromBuff();
  std::vector<cv::Mat> images;
  in_imgVec.push_back(color);
  in_imgVec.push_back(depth);
}

void TemplateGenerator::printProgBar(uint16_t in_percent,
                                     std::string const& in_mfile) {
  std::string bar;

  for (int i = 0; i < 50; i++) {
    if (i < (in_percent / 2)) {
      bar.replace(i, 1, "=");
    } else if (i == (in_percent / 2)) {
      bar.replace(i, 1, ">");
    } else {
      bar.replace(i, 1, " ");
    }
  }
  std::cout << "\r"
               "["
            << bar << "] ";
  std::cout.width(3);
  std::cout << in_percent << "%     " + in_mfile << std::flush;
  if (in_percent == 100) {
    std::cout << std::endl;
  }
}

uint16_t TemplateGenerator::calculateCurrentPercent(
    uint16_t const& in_spehreRadius, uint16_t const& in_currentIteration) {
  uint16_t numberOfDiffRadius =
      std::floor((endDistance_ - startDistance_ + stepSize_) / stepSize_);
  return std::round((float)(in_currentIteration + 1) * 100.0f /
                        (float)numCameraVertices_ / (float)numberOfDiffRadius +
                    (float)(in_spehreRadius - startDistance_) /
                        (float)stepSize_ * 100 / (float)numberOfDiffRadius);
}

void TemplateGenerator::writeSettings() {
  CameraParameters camParam = CameraParameters();
  TemplateGenerationSettings temp = TemplateGenerationSettings();
  std::string filename = "linemod_settings.yml";
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs.writeComment("###### CAMERA PARAMETERS ######");
  fs << "video width" << camParam.videoWidth;
  fs << "video height" << camParam.videoHeight;
  fs << "camera fx" << camParam.fx;
  fs << "camera fy" << camParam.fy;
  fs << "camera cx" << camParam.cx;
  fs << "camera cy" << camParam.cy;
  fs << "distortion parameters" << camParam.distortionCoefficients;
  fs << "color" << cv::Scalar(0, 20, 100);
  fs.writeComment("###### TEMPLATE GENERATION SETTINGS ######");
  fs << "model folder" << temp.modelFolder;
  fs << "model file ending" << temp.modelFileEnding;
  fs << "only use color modality" << temp.onlyUseColorModality;
  fs << "in plane rotation starting angle" << temp.angleStart;
  fs << "in plane rotation stopping angle" << temp.angleStop;
  fs << "in plane rotation angle step" << temp.angleStep;
  fs << "distance start" << temp.startDistance;
  fs << "distance stop" << temp.endDistance;
  fs << "distance step" << temp.stepSize;
  fs << "icosahedron subdivisions" << temp.subdivisions;
}
