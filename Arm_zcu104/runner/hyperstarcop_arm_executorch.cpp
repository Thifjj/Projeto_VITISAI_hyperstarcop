#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/platform/runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using executorch::extension::Module;
using executorch::extension::from_blob;

constexpr int kH = 512;
constexpr int kW = 512;
constexpr int kC = 4;
constexpr size_t kPixels = static_cast<size_t>(kH) * kW;
constexpr size_t kInputElements = kPixels * kC;

struct Options {
  std::string profile = "all";
  std::string model = "/home/root/hyperstarcop_xnnpack_fp32.pte";
  std::string dataset = "/home/root/STARCOP_mini";
  std::string csv = "/home/root/STARCOP_mini/test_mini10.csv";
  std::string out = "/home/root/hyperstarcop_arm_results";
  int threads = 2;
  int warmup = 3;
  int model_only_iterations = 20;
  int end_to_end_passes = 3;
  bool validate = true;
};

struct DatasetItem {
  std::string id;
  std::string has_plume;
};

struct Counts {
  uint64_t tp = 0, fp = 0, fn = 0, tn = 0;
};

struct Metrics {
  double precision = 0, recall = 0, f1 = 0, iou = 0, accuracy = 0;
};

struct Stats {
  size_t count = 0;
  double mean = 0, median = 0, p90 = 0, p95 = 0, p99 = 0;
  double min = 0, max = 0, stddev = 0, cv = 0;
};

struct Sample {
  std::string mode;
  size_t job = 0, dataset_index = 0;
  double io_ms = 0, preprocess_ms = 0, inference_ms = 0;
  double postprocess_ms = 0, latency_ms = 0, completion_s = 0;
};

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void usage(const char* program) {
  std::cout
      << "Uso: " << program << " [opcoes]\n"
      << "  --profile all|validation|model-only|end-to-end\n"
      << "  --model PATH --dataset DIR --csv PATH --out DIR\n"
      << "  --threads 1|2|4 --warmup N\n"
      << "  --model-only-iterations N --end-to-end-passes N\n"
      << "  --validate | --no-validate\n";
}

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("Valor ausente para " + arg);
      return argv[i];
    };
    if (arg == "--profile") o.profile = value();
    else if (arg == "--model") o.model = value();
    else if (arg == "--dataset") o.dataset = value();
    else if (arg == "--csv") o.csv = value();
    else if (arg == "--out") o.out = value();
    else if (arg == "--threads") o.threads = std::stoi(value());
    else if (arg == "--warmup") o.warmup = std::stoi(value());
    else if (arg == "--model-only-iterations") {
      o.model_only_iterations = std::stoi(value());
    } else if (arg == "--end-to-end-passes") {
      o.end_to_end_passes = std::stoi(value());
    } else if (arg == "--validate") o.validate = true;
    else if (arg == "--no-validate") o.validate = false;
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Argumento desconhecido: " + arg);
    }
  }
  const std::vector<std::string> profiles{
      "all", "validation", "model-only", "end-to-end"};
  if (std::find(profiles.begin(), profiles.end(), o.profile) == profiles.end()) {
    throw std::runtime_error("Perfil invalido: " + o.profile);
  }
  if (o.threads < 1 || o.warmup < 0 || o.model_only_iterations < 1 ||
      o.end_to_end_passes < 1) {
    throw std::runtime_error("Opcao numerica invalida");
  }
  return o;
}

std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string current;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        current += '"';
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(current);
      current.clear();
    } else if (ch != '\r') {
      current += ch;
    }
  }
  fields.push_back(current);
  return fields;
}

std::vector<DatasetItem> load_dataset_csv(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Nao foi possivel abrir CSV: " + path);
  std::string line;
  if (!std::getline(file, line)) throw std::runtime_error("CSV vazio");
  const auto header = parse_csv_line(line);
  int id_col = -1, plume_col = -1;
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == "id") id_col = static_cast<int>(i);
    if (header[i] == "has_plume") plume_col = static_cast<int>(i);
  }
  if (id_col < 0) throw std::runtime_error("CSV sem coluna id");
  std::vector<DatasetItem> items;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    const auto fields = parse_csv_line(line);
    if (static_cast<size_t>(id_col) >= fields.size()) continue;
    DatasetItem item{fields[id_col], ""};
    if (plume_col >= 0 && static_cast<size_t>(plume_col) < fields.size()) {
      item.has_plume = fields[plume_col];
    }
    items.push_back(std::move(item));
  }
  if (items.empty()) throw std::runtime_error("CSV sem imagens");
  return items;
}

cv::Mat read_tif_float(const fs::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
  if (image.empty()) throw std::runtime_error("Falha ao ler: " + path.string());
  if (image.channels() != 1 || image.rows != kH || image.cols != kW) {
    throw std::runtime_error("TIFF deve ser monocanal 512x512: " + path.string());
  }
  cv::Mat output;
  image.convertTo(output, CV_32F);
  return output;
}

inline float clip02(float value) {
  return std::max(0.0f, std::min(2.0f, value));
}

void load_and_preprocess(
    const fs::path& folder,
    std::vector<float>& input,
    double& io_ms,
    double& preprocess_ms) {
  const auto io_begin = Clock::now();
  const cv::Mat mag1c = read_tif_float(folder / "mag1c.tif");
  const cv::Mat red = read_tif_float(folder / "TOA_AVIRIS_640nm.tif");
  const cv::Mat green = read_tif_float(folder / "TOA_AVIRIS_550nm.tif");
  const cv::Mat blue = read_tif_float(folder / "TOA_AVIRIS_460nm.tif");
  const auto io_end = Clock::now();

  const auto norm_begin = Clock::now();
  input.resize(kInputElements);
  for (int y = 0; y < kH; ++y) {
    const float* rows[4] = {
        mag1c.ptr<float>(y), red.ptr<float>(y),
        green.ptr<float>(y), blue.ptr<float>(y)};
    const float divisors[4] = {1750.0f, 60.0f, 60.0f, 60.0f};
    for (int x = 0; x < kW; ++x) {
      const size_t pixel = static_cast<size_t>(y) * kW + x;
      for (int c = 0; c < kC; ++c) {
        input[static_cast<size_t>(c) * kPixels + pixel] =
            clip02(rows[c][x] / divisors[c]);
      }
    }
  }
  const auto norm_end = Clock::now();
  io_ms = elapsed_ms(io_begin, io_end);
  preprocess_ms = io_ms + elapsed_ms(norm_begin, norm_end);
}

cv::Mat load_label(const fs::path& folder) {
  const cv::Mat source = read_tif_float(folder / "labelbinary.tif");
  cv::Mat label(kH, kW, CV_8U);
  for (int y = 0; y < kH; ++y) {
    const float* src = source.ptr<float>(y);
    uint8_t* dst = label.ptr<uint8_t>(y);
    for (int x = 0; x < kW; ++x) dst[x] = src[x] > 0.0f ? 1 : 0;
  }
  return label;
}

class Engine {
 public:
  Engine(const std::string& model_path, int threads)
      : module_(model_path, Module::LoadMode::MmapUseMadvise) {
    executorch::runtime::runtime_init();
    auto* pool = executorch::extension::threadpool::get_threadpool();
    if (pool == nullptr || !pool->_unsafe_reset_threadpool(threads)) {
      throw std::runtime_error("Falha ao configurar threadpool");
    }
    const auto error = module_.load_forward();
    if (error != executorch::runtime::Error::Ok) {
      throw std::runtime_error("Falha ao carregar forward do PTE");
    }
  }

  double infer(const std::vector<float>& input, std::vector<float>* output) {
    auto tensor = from_blob(
        const_cast<float*>(input.data()), {1, kC, kH, kW});
    const auto begin = Clock::now();
    auto result = module_.forward(tensor);
    const auto end = Clock::now();
    if (!result.ok()) throw std::runtime_error("Falha durante forward");
    if (result->size() != 1 || !result->at(0).isTensor()) {
      throw std::runtime_error("Saida inesperada do modelo");
    }
    const auto out_tensor = result->at(0).toTensor();
    if (out_tensor.numel() != kPixels) {
      throw std::runtime_error("Saida nao possui 512x512 elementos");
    }
    if (output != nullptr) {
      const float* data = out_tensor.const_data_ptr<float>();
      output->assign(data, data + kPixels);
    }
    return elapsed_ms(begin, end);
  }

 private:
  Module module_;
};

void postprocess(const std::vector<float>& logits, std::vector<uint8_t>& mask) {
  mask.resize(kPixels);
  for (size_t i = 0; i < kPixels; ++i) {
    const float z = std::max(-80.0f, std::min(80.0f, logits[i]));
    const float probability = 1.0f / (1.0f + std::exp(-z));
    mask[i] = probability > 0.5f ? 1 : 0;
  }
}

Counts confusion(const std::vector<uint8_t>& prediction, const cv::Mat& label) {
  Counts counts;
  for (int y = 0; y < kH; ++y) {
    const uint8_t* truth = label.ptr<uint8_t>(y);
    for (int x = 0; x < kW; ++x) {
      const bool pred = prediction[static_cast<size_t>(y) * kW + x] != 0;
      const bool real = truth[x] != 0;
      if (pred && real) ++counts.tp;
      else if (pred) ++counts.fp;
      else if (real) ++counts.fn;
      else ++counts.tn;
    }
  }
  return counts;
}

Metrics calculate_metrics(const Counts& c) {
  Metrics m;
  if (c.tp + c.fp) m.precision = double(c.tp) / double(c.tp + c.fp);
  if (c.tp + c.fn) m.recall = double(c.tp) / double(c.tp + c.fn);
  if (m.precision + m.recall) {
    m.f1 = 2.0 * m.precision * m.recall / (m.precision + m.recall);
  }
  if (c.tp + c.fp + c.fn) m.iou = double(c.tp) / double(c.tp + c.fp + c.fn);
  const uint64_t total = c.tp + c.fp + c.fn + c.tn;
  if (total) m.accuracy = double(c.tp + c.tn) / double(total);
  return m;
}

Stats summarize(std::vector<double> values) {
  Stats s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.count = values.size();
  s.min = values.front();
  s.max = values.back();
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  auto percentile = [&](double p) {
    const double position = p * (values.size() - 1);
    const size_t index = static_cast<size_t>(position);
    const double fraction = position - index;
    return values[index] * (1.0 - fraction) +
        values[std::min(index + 1, values.size() - 1)] * fraction;
  };
  s.median = percentile(0.50);
  s.p90 = percentile(0.90);
  s.p95 = percentile(0.95);
  s.p99 = percentile(0.99);
  double squared = 0;
  for (double value : values) squared += (value - s.mean) * (value - s.mean);
  s.stddev = std::sqrt(squared / values.size());
  s.cv = s.mean ? s.stddev / s.mean : 0;
  return s;
}

void write_metrics(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    Engine& engine) {
  std::ofstream per_image(fs::path(o.out) / "metricas_por_imagem.csv");
  per_image << "id,has_plume,TP,FP,FN,TN,precision,recall,f1,iou,accuracy\n";
  Counts global;
  std::vector<float> input, logits;
  std::vector<uint8_t> mask;
  for (size_t i = 0; i < dataset.size(); ++i) {
    const fs::path folder = fs::path(o.dataset) / dataset[i].id;
    double io = 0, preprocess = 0;
    load_and_preprocess(folder, input, io, preprocess);
    engine.infer(input, &logits);
    postprocess(logits, mask);
    const Counts c = confusion(mask, load_label(folder));
    const Metrics m = calculate_metrics(c);
    global.tp += c.tp; global.fp += c.fp; global.fn += c.fn; global.tn += c.tn;
    per_image << dataset[i].id << ',' << dataset[i].has_plume << ','
              << c.tp << ',' << c.fp << ',' << c.fn << ',' << c.tn << ','
              << std::setprecision(12) << m.precision << ',' << m.recall << ','
              << m.f1 << ',' << m.iou << ',' << m.accuracy << '\n';
    std::cout << "[VAL " << i + 1 << '/' << dataset.size() << "] "
              << dataset[i].id << " F1=" << std::fixed << std::setprecision(4)
              << m.f1 << " IoU=" << m.iou << '\n';
  }
  const Metrics m = calculate_metrics(global);
  std::ofstream summary(fs::path(o.out) / "metricas_globais.csv");
  summary << "num_imagens,TP,FP,FN,TN,precision_global,recall_global,"
             "f1_global,iou_global,accuracy_global\n";
  summary << dataset.size() << ',' << global.tp << ',' << global.fp << ','
          << global.fn << ',' << global.tn << ',' << std::setprecision(12)
          << m.precision << ',' << m.recall << ',' << m.f1 << ',' << m.iou
          << ',' << m.accuracy << '\n';
  std::cout << "GLOBAL Precision=" << m.precision << " Recall=" << m.recall
            << " F1=" << m.f1 << " IoU=" << m.iou
            << " Accuracy=" << m.accuracy << '\n';
}

void append_samples(const fs::path& path, const std::vector<Sample>& samples) {
  std::ofstream file(path, std::ios::app);
  if (file.tellp() == 0) {
    file << "mode,job,dataset_index,io_ms,preprocess_ms,inference_ms,"
            "postprocess_ms,e2e_ms,completion_s,instantaneous_fps\n";
  }
  file << std::setprecision(12);
  for (const auto& s : samples) {
    file << s.mode << ',' << s.job << ',' << s.dataset_index << ',' << s.io_ms
         << ',' << s.preprocess_ms << ',' << s.inference_ms << ','
         << s.postprocess_ms << ',' << s.latency_ms << ',' << s.completion_s
         << ',' << (s.latency_ms ? 1000.0 / s.latency_ms : 0.0) << '\n';
  }
}

void append_summary(
    const fs::path& path,
    const std::string& mode,
    int threads,
    double wall_s,
    const std::vector<Sample>& samples) {
  std::vector<double> latency, inference, preprocess, postprocess;
  for (const auto& s : samples) {
    latency.push_back(s.latency_ms);
    inference.push_back(s.inference_ms);
    preprocess.push_back(s.preprocess_ms);
    postprocess.push_back(s.postprocess_ms);
  }
  const Stats l = summarize(latency), inf = summarize(inference);
  const Stats pre = summarize(preprocess), post = summarize(postprocess);
  const double fps = samples.size() / wall_s;
  std::ofstream file(path, std::ios::app);
  if (file.tellp() == 0) {
    file << "mode,batch,threads,completed,wall_s,throughput_fps,"
            "equiv_fps_avg,latency_mean_ms,latency_median_ms,latency_min_ms,"
            "latency_max_ms,latency_p90_ms,latency_p95_ms,latency_p99_ms,"
            "latency_stddev_ms,latency_cv,inference_mean_ms,inference_p95_ms,"
            "inference_p99_ms,preprocess_mean_ms,postprocess_mean_ms\n";
  }
  file << std::setprecision(12) << mode << ",1," << threads << ','
       << samples.size() << ',' << wall_s << ',' << fps << ','
       << (l.mean ? 1000.0 / l.mean : 0.0) << ',' << l.mean << ',' << l.median
       << ',' << l.min << ',' << l.max << ',' << l.p90 << ',' << l.p95 << ','
       << l.p99 << ',' << l.stddev << ',' << l.cv << ',' << inf.mean << ','
       << inf.p95 << ',' << inf.p99 << ',' << pre.mean << ',' << post.mean
       << '\n';
  std::cout << "\n" << mode << " threads=" << threads
            << " completed=" << samples.size() << " wall=" << wall_s
            << " s throughput=" << fps << " FPS\n"
            << "latency mean=" << l.mean << " ms p95=" << l.p95
            << " ms p99=" << l.p99 << " ms\n";
}

std::vector<Sample> benchmark_model_only(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    Engine& engine) {
  std::vector<float> input;
  double io = 0, preprocess = 0;
  load_and_preprocess(fs::path(o.dataset) / dataset.front().id,
                      input, io, preprocess);
  for (int i = 0; i < o.warmup; ++i) engine.infer(input, nullptr);
  std::vector<Sample> samples;
  samples.reserve(o.model_only_iterations);
  const auto wall_begin = Clock::now();
  for (int i = 0; i < o.model_only_iterations; ++i) {
    Sample sample;
    sample.mode = "baseline_model_only";
    sample.job = i;
    sample.dataset_index = i % dataset.size();
    sample.inference_ms = engine.infer(input, nullptr);
    sample.latency_ms = sample.inference_ms;
    sample.completion_s = std::chrono::duration<double>(
        Clock::now() - wall_begin).count();
    samples.push_back(sample);
  }
  const double wall_s = std::chrono::duration<double>(
      Clock::now() - wall_begin).count();
  append_summary(fs::path(o.out) / "benchmark_summary.csv",
                 "baseline_model_only", o.threads, wall_s, samples);
  return samples;
}

std::vector<Sample> benchmark_end_to_end(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    Engine& engine) {
  std::vector<float> input, logits;
  std::vector<uint8_t> mask;
  // One complete untimed pass warms model, decoder and filesystem cache,
  // matching the existing DPU baseline methodology.
  for (size_t i = 0; i < dataset.size(); ++i) {
    double io = 0, preprocess = 0;
    load_and_preprocess(fs::path(o.dataset) / dataset[i].id,
                        input, io, preprocess);
    engine.infer(input, &logits);
    postprocess(logits, mask);
  }
  std::vector<Sample> samples;
  samples.reserve(static_cast<size_t>(o.end_to_end_passes) * dataset.size());
  const auto wall_begin = Clock::now();
  size_t job = 0;
  for (int pass = 0; pass < o.end_to_end_passes; ++pass) {
    for (size_t i = 0; i < dataset.size(); ++i, ++job) {
      Sample sample;
      sample.mode = "baseline_end_to_end";
      sample.job = job;
      sample.dataset_index = i;
      const auto begin = Clock::now();
      load_and_preprocess(fs::path(o.dataset) / dataset[i].id,
                          input, sample.io_ms, sample.preprocess_ms);
      sample.inference_ms = engine.infer(input, &logits);
      const auto post_begin = Clock::now();
      postprocess(logits, mask);
      const auto end = Clock::now();
      sample.postprocess_ms = elapsed_ms(post_begin, end);
      sample.latency_ms = elapsed_ms(begin, end);
      sample.completion_s = std::chrono::duration<double>(
          end - wall_begin).count();
      samples.push_back(sample);
    }
  }
  const double wall_s = std::chrono::duration<double>(
      Clock::now() - wall_begin).count();
  append_summary(fs::path(o.out) / "benchmark_summary.csv",
                 "baseline_end_to_end", o.threads, wall_s, samples);
  return samples;
}

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (!fs::is_regular_file(options.model)) {
      throw std::runtime_error("PTE nao encontrado: " + options.model);
    }
    fs::create_directories(options.out);
    const auto dataset = load_dataset_csv(options.csv);
    std::cout << "HyperSTARCOP ExecuTorch/XNNPACK ARM\n"
              << "imagens=" << dataset.size() << " threads=" << options.threads
              << " batch=1\n";
    Engine engine(options.model, options.threads);
    if (options.validate &&
        (options.profile == "all" || options.profile == "validation")) {
      write_metrics(options, dataset, engine);
    }
    std::vector<Sample> all_samples;
    if (options.profile == "all" || options.profile == "model-only") {
      auto samples = benchmark_model_only(options, dataset, engine);
      all_samples.insert(all_samples.end(), samples.begin(), samples.end());
    }
    if (options.profile == "all" || options.profile == "end-to-end") {
      auto samples = benchmark_end_to_end(options, dataset, engine);
      all_samples.insert(all_samples.end(), samples.begin(), samples.end());
    }
    append_samples(fs::path(options.out) / "benchmark_samples.csv", all_samples);
    std::cout << "Resultados: " << options.out << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERRO: " << error.what() << '\n';
    return 1;
  }
}
