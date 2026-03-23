<!--
 * @Author: juling julinger@qq.com
 * @Date: 2026-03-23 11:25:14
 * @LastEditors: juling julinger@qq.com
 * @LastEditTime: 2026-03-23 14:09:00
-->
>note: Line-Mod源码理解，基于opencv4.2.0+opencv_contrib，位于rgbd子模块。

# 一、头文件
linemod.hpp头文件主要结构，外部只有以下接口开放，其它功能函数定义在cpp内部。

```c++
struct Feature {}; // omit
struct Template{}; // omit

class QuantizedPyramid {
public:
  virtual ~QuantizedPyramid() {}
  virtual void quantize(CV_OUT Mat& dst) const =0;
  virtual bool extractTemplate(CV_OUT Template& templ) const =0;
  virtual void pyrDown() =0;
protected:
  struct Candidate{}; // omit
  static void selectScatteredFeatures(const std::vector<Candidate>&candidates,
                                      std::vector<Feature>& features,
                                      size_t num_features, float distance);
};

// 模态基类
class Modality{
public:
  virtual ~Modality() {}
  Ptr<QuantizedPyramid> process(const Mat& src, 
                                const Mat& mask = Mat()) const  
                                { return processImpl(src, mask);}
  virtual String name() const =0;
  virtual void read(const FileNode& fn) =0;
  virtual void write(FileStorage& fs) const =0;
  static Ptr<Modality> create(const String& modality_type);
  static Ptr<Modality> create(const FileNode& fn);
protected:
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src, 
                                            const Mat& mask) const =0;
};

// 梯度模态class
class ColorGradient : public Modality {
public:
  ColorGradient();
  ColorGradient(float weak_threshold, size_t num_features, 
                float strong_threshold);
  static Ptr<ColorGradient> create(float weak_threshold, 
                                   size_t num_features,  
                                   float strong_threshold);
  virtual String name() const CV_OVERRIDE;
  virtual void read(const FileNode& fn) CV_OVERRIDE;
  virtual void write(FileStorage& fs) const CV_OVERRIDE;
  
  float weak_threshold;
  size_t num_features;
  float strong_threshold;

protected:
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
                        const Mat& mask) const CV_OVERRIDE;
}；

// 法向量模态class
class DepthNormal : public Modality {
public:
  DepthNormal();
  DepthNormal(int distance_threshold, int difference_threshold, 
              size_t num_features, int extract_threshold);
  static Ptr<DepthNormal> create(int distance_threshold, 
                                 int difference_threshold,
                                 size_t num_features, 
                                 int extract_threshold);
  virtual String name() const CV_OVERRIDE;
  virtual void read(const FileNode& fn) CV_OVERRIDE;
  virtual void write(FileStorage& fs) const CV_OVERRIDE;

  int distance_threshold;
  int difference_threshold;
  size_t num_features;
  int extract_threshold;

protected:
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
                        const Mat& mask) const CV_OVERRIDE;
};

void colormap(const Mat& quantized, CV_OUT Mat& dst);
void drawFeatures(InputOutputArray img, 
                  const std::vector<Template>& templates, 
                  const Point2i& tl, int size = 10);
struct Match{}; // omit

// 检测器class
class Detector {
public:
  Detector();
  Detector(const std::vector<Ptr<Modality>>& modalities, 
           const std::vector<int>& T_pyramid);
  void match(const std::vector<Mat>& sources, float threshold, 
            CV_OUT std::vector<Match>& matches,
            const std::vector<String>& class_ids = std::vector<String>(),
            OutputArrayOfArrays quantized_images = noArray(),
            const std::vector<Mat>& masks = std::vector<Mat>()) const;
  int addTemplate(const std::vector<Mat>& sources, const String& class_id,
                  const Mat& object_mask, CV_OUT Rect* bounding_box = NULL);
  int addSyntheticTemplate(const std::vector<Template>& templates, 
                           const String& class_id);
  const std::vector<Ptr<Modality>>& getModalities() const 
                                    { return modalities; }
  int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }
  int pyramidLevels() const { return pyramid_levels; }
  const std::vector<Template>& getTemplates(const String& class_id, 
                                            int template_id) const;
  int numTemplates() const;
  int numTemplates(const String& class_id) const;
  int numClasses() const { return static_cast<int>(class_templates.size()); }
  std::vector<String> classIds() const;
  void read(const FileNode& fn);
  void write(FileStorage& fs) const;
  String readClass(const FileNode& fn, const String &class_id_override = "");
  void writeClass(const String& class_id, FileStorage& fs) const;
  void readClasses(const std::vector<String>& class_ids,
                   const String& format = "templates_%s.yml.gz");
  void writeClasses(const String& format = "templates_%s.yml.gz") const;

protected:
  std::vector< Ptr<Modality> > modalities;
  int pyramid_levels;
  std::vector<int> T_at_level;

  typedef std::vector<Template> TemplatePyramid;
  typedef std::map<String, std::vector<TemplatePyramid>> TemplatesMap;
  TemplatesMap class_templates;

  typedef std::vector<Mat> LinearMemories;
  typedef std::vector< std::vector<LinearMemories> > LinearMemoryPyramid;

  void matchClass(const LinearMemoryPyramid& lm_pyramid,
                  const std::vector<Size>& sizes,
                  float threshold, std::vector<Match>& matches,
                  const String& class_id,
                  const std::vector<TemplatePyramid>& template_pyramids) const;
};

Ptr<linemod::Detector> getDefaultLINE();
Ptr<linemod::Detector> getDefaultLINEMOD();
```

# 二、cpp解读
## 1. 整体结构
```c++
// cpp内部使用
static inline int getLabel(int quantized) {}

// Feature结构体中函数的implement
void Feature::read(const FileNode& fn) {}
void Feature::write(FileStorage& fs) const {}

// cpp内部使用
static Rect cropTemplates(std::vector<Template>& templates) {}

// Template结构体中函数的implement
void Template::read(const FileNode& fn){}
void Template::write(FileStorage& fs) const {}

// QuantizedPyramid其它函数均为纯虚函数，由子类重写，只有selectScatteredFeatures为基类实现
void QuantizedPyramid::selectScatteredFeatures(const std::vector<Candidate>&candidates,
                                               std::vector<Feature>& features,
                                               size_t num_features, float distance) {}

// Modality类，只有create需要实现，其它均为纯虚函数，由子类实现。
// process为公共接口，返回processImpl调用结果，processImpl为protected的纯虚函数。
Ptr<Modality> Modality::create(const String& modality_type){} 
Ptr<Modality> Modality::create(const FileNode& fn) {}

// 普通全局函数，头文件中已declaration，此处implement
void colormap(const Mat& quantized, Mat& dst){}
void drawFeatures(InputOutputArray img, 
                  const std::vector<Template>& templates,
                  const Point2i& tl, int size){}

// Forward declaration
void hysteresisGradient(Mat& magnitude, Mat& angle,
                        Mat& ap_tmp, float threshold); 
// cpp内部使用
static void quantizedOrientations(const Mat& src, Mat& magnitude,
                           Mat& angle, float threshold) {} 
// 此处implement
void hysteresisGradient(Mat& magnitude, Mat& quantized_angle,
                        Mat& angle, float threshold) {}

// 梯度量化金字塔class declaration
class ColorGradientPyramid : public QuantizedPyramid {
public:
  ColorGradientPyramid(const Mat& src, const Mat& mask,
                       float weak_threshold, size_t num_features,
                       float strong_threshold);
  virtual void quantize(Mat& dst) const CV_OVERRIDE;
  virtual bool extractTemplate(Template& templ) const CV_OVERRIDE;
  virtual void pyrDown() CV_OVERRIDE;
protected:
  void update(); // Recalculate angle and magnitude images

  Mat src;
  Mat mask;

  int pyramid_level;
  Mat angle;
  Mat magnitude;

  float weak_threshold;
  size_t num_features;
  float strong_threshold;
}；

// 梯度量化金字塔implement
ColorGradientPyramid::ColorGradientPyramid(const Mat& _src, const Mat& _mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold) {}
void ColorGradientPyramid::update() {}
void ColorGradientPyramid::pyrDown() {}
void ColorGradientPyramid::quantize(Mat& dst) const {}
bool ColorGradientPyramid::extractTemplate(Template& templ) const {}

// 梯度模态implement
ColorGradient::ColorGradient() {}
ColorGradient::ColorGradient(float _weak_threshold, 
                             size_t _num_features, 
                             float _strong_threshold)
  : weak_threshold(_weak_threshold),
    num_features(_num_features),
    strong_threshold(_strong_threshold) {}
Ptr<ColorGradient> ColorGradient::create(float weak_threshold, size_t num_features, 
                                         float strong_threshold) {}
static const char CG_NAME[] = "ColorGradient";
String ColorGradient::name() const { return CG_NAME; };

Ptr<QuantizedPyramid> ColorGradient::processImpl(const Mat& src,
                                                 const Mat& mask) const {
  return makePtr<ColorGradientPyramid>(src, mask, weak_threshold, 
                                       num_features, strong_threshold);
}
void ColorGradient::read(const FileNode& fn) {}
void ColorGradient::write(FileStorage& fs) const {}

// 法向量模态implement
#include "normal_lut.i"
static void accumBilateral(long delta, long i, long j, long * A, long * b, int threshold) {}
static void quantizedNormals(const Mat& src, Mat& dst, int distance_threshold,
                      int difference_threshold) {}

// 法向量量化金字塔class declaration
class DepthNormalPyramid : public QuantizedPyramid
{
public:
  DepthNormalPyramid(const Mat& src, const Mat& mask,
                     int distance_threshold, int difference_threshold, size_t num_features,
                     int extract_threshold);

  virtual void quantize(Mat& dst) const CV_OVERRIDE;

  virtual bool extractTemplate(Template& templ) const CV_OVERRIDE;

  virtual void pyrDown() CV_OVERRIDE;

protected:
  Mat mask;

  int pyramid_level;
  Mat normal;

  size_t num_features;
  int extract_threshold;
};

// 法向量量化金字塔implement
DepthNormalPyramid::DepthNormalPyramid(const Mat& src, const Mat& _mask,
                                       int distance_threshold, int difference_threshold, size_t _num_features,
                                       int _extract_threshold)
  : mask(_mask),
    pyramid_level(0),
    num_features(_num_features),
    extract_threshold(_extract_threshold) {}
void DepthNormalPyramid::pyrDown() {}
void DepthNormalPyramid::quantize(Mat& dst) const {}
bool DepthNormalPyramid::extractTemplate(Template& templ) const {}

// 法向量模态implement
DepthNormal::DepthNormal() : distance_threshold(2000),
                            difference_threshold(50),
                            num_features(63),
                            extract_threshold(2) {}
Ptr<DepthNormal> DepthNormal::create(int distance_threshold, int difference_threshold, 
                                     size_t num_features, int extract_threshold) {}
static const char DN_NAME[] = "DepthNormal";
String DepthNormal::name() const { return DN_NAME; }
Ptr<QuantizedPyramid> DepthNormal::processImpl(const Mat& src,
                                               const Mat& mask) const{
  return makePtr<DepthNormalPyramid>(src, mask, distance_threshold, difference_threshold,
                                     num_features, extract_threshold);
}
void DepthNormal::read(const FileNode& fn) {}
void DepthNormal::write(FileStorage& fs) const {}

// 一些cpp内部使用的函数implement
static void orUnaligned8u(const uchar * src, const int src_stride,
                   uchar * dst, const int dst_stride,
                   const int width, const int height) {}
static void spread(const Mat& src, Mat& dst, int T) {}
static void computeResponseMaps(const Mat& src, std::vector<Mat>& response_maps) {}
static void linearize(const Mat& response_map, Mat& linearized, int T) {}
static const unsigned char* accessLinearMemory(const std::vector<Mat>& linear_memories,
          const Feature& f, int T, int W) {}
static void similarity(const std::vector<Mat>& linear_memories, const Template& templ,
                Mat& dst, Size size, int T) {}
static void similarityLocal(const std::vector<Mat>& linear_memories, const Template& templ,
                     Mat& dst, Size size, int T, Point center) {}
static void addUnaligned8u16u(const uchar* src1, const uchar* src2, ushort* res, int length) {}
static void addSimilarities(const std::vector<Mat>& similarities, Mat& dst) {}

// 检测器implement
Detector::Detector() {}
Detector::Detector(const std::vector< Ptr<Modality> >& _modalities,
                   const std::vector<int>& T_pyramid)
  : modalities(_modalities),
    pyramid_levels(static_cast<int>(T_pyramid.size())),
    T_at_level(T_pyramid) {}
void Detector::match(const std::vector<Mat>& sources, float threshold, 
                     std::vector<Match>& matches, 
                     const std::vector<String>& class_ids, 
                     OutputArrayOfArrays quantized_images, 
                     const std::vector<Mat>& masks) const {}
// cpp内部使用的结构体
struct MatchPredicate
{
  MatchPredicate(float _threshold) : threshold(_threshold) {}
  bool operator() (const Match& m) { return m.similarity < threshold; }
  float threshold;
};
void Detector::matchClass(const LinearMemoryPyramid& lm_pyramid,
                          const std::vector<Size>& sizes,
                          float threshold, std::vector<Match>& matches,
                          const String& class_id,
                          const std::vector<TemplatePyramid>& template_pyramids) const {}
int Detector::addTemplate(const std::vector<Mat>& sources, const String& class_id,
                          const Mat& object_mask, Rect* bounding_box) {}
int Detector::addSyntheticTemplate(const std::vector<Template>& templates, 
                                   const String& class_id) {}
const std::vector<Template>& Detector::getTemplates(const String& class_id, 
                                                    int template_id) const {}
int Detector::numTemplates() const {}
int Detector::numTemplates(const String& class_id) const {}
std::vector<String> Detector::classIds() const {}
void Detector::read(const FileNode& fn) {}
void Detector::write(FileStorage& fs) const {}
String Detector::readClass(const FileNode& fn, const String &class_id_override) {}
void Detector::writeClass(const String& class_id, FileStorage& fs) const {}
void Detector::readClasses(const std::vector<String>& class_ids,
                           const String& format) {}
void Detector::writeClasses(const String& format) const {}

// cpp内部使用
static const int T_DEFAULTS[] = {5, 8};

// 头文件中已declaration，此处implement
Ptr<Detector> getDefaultLINE()
{
  std::vector< Ptr<Modality> > modalities;
  modalities.push_back(makePtr<ColorGradient>());
  return makePtr<Detector>(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}

Ptr<Detector> getDefaultLINEMOD()
{
  std::vector< Ptr<Modality> > modalities;
  modalities.push_back(makePtr<ColorGradient>());
  modalities.push_back(makePtr<DepthNormal>());
  return makePtr<Detector>(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}
```


