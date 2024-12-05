#ifndef FILTERDECODER_CXX
#define FILTERDECODER_CXX

#include "DecoderToolBase.h"

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include <torch/torch.h>

using anab::FeatureVector;
using anab::MVADescription;

class FilterDecoder : public DecoderToolBase
{

public:

  /**
   *  @brief  Constructor
   *
   *  @param  pset
   */
  FilterDecoder(const fhicl::ParameterSet &pset);

  /**
   *  @brief  Virtual Destructor
   */
  virtual ~FilterDecoder() noexcept = default;
    
  /**
   *  @brief Interface for configuring the particular algorithm tool
   *
   *  @param ParameterSet  The input set of parameters for configuration
   */
  void configure(const fhicl::ParameterSet&);

  /**
   * @brief declareProducts function
   *
   */
  std::pair<DecoderType,std::string> declareProducts() override { return std::make_pair(Filter,"filter"); };

  /**
   * @brief writeEmptyToEvent function
   *
   * @param art::Event event record
   */
  void writeEmptyToEvent(art::Event& e, const vector<vector<size_t> >& idsmap) override;

  /**
   * @brief Decoder function
   *
   * @param art::Event event record for decoder
   */
  void writeToEvent(art::Event& e, const vector<vector<size_t> >& idsmap, const vector<NuGraphOutput>& infer_output) override;

};

FilterDecoder::FilterDecoder(const fhicl::ParameterSet &p)
{
  configure(p);
}

void FilterDecoder::configure(const fhicl::ParameterSet& p) {}

void FilterDecoder::writeEmptyToEvent(art::Event& e, const vector<vector<size_t> >& idsmap) {
  //
  size_t size = 0;
  for (auto& v : idsmap) size += v.size();
  std::unique_ptr<vector<FeatureVector<1>>> filtcol(new vector<FeatureVector<1>>(size, FeatureVector<1>(std::array<float, 1>({-1.}))));
  e.put(std::move(filtcol), "filter");
  //
}

void FilterDecoder::writeToEvent(art::Event& e, const vector<vector<size_t> >& idsmap, const vector<NuGraphOutput>& infer_output) {
  //
  size_t size = 0;
  for (auto& v : idsmap) size += v.size();
  std::unique_ptr<vector<FeatureVector<1>>> filtcol(new vector<FeatureVector<1>>(size, FeatureVector<1>(std::array<float, 1>({-1.}))));

  std::vector<float> x_filter_u_data;
  std::vector<float> x_filter_v_data;
  std::vector<float> x_filter_y_data;

  for (auto& io : infer_output) {
    if (io.output_name == "x_filter_u") x_filter_u_data = io.output_vec;
    if (io.output_name == "x_filter_v") x_filter_v_data = io.output_vec;
    if (io.output_name == "x_filter_y") x_filter_y_data = io.output_vec;
  }

  if (debug) {
    std::cout << "x_filter_u: " << std::endl;
    printVector(x_filter_u_data);

    std::cout << "x_filter_v: " << std::endl;
    printVector(x_filter_v_data);

    std::cout << "x_filter_y: " << std::endl;
    printVector(x_filter_y_data);
  }

  for (size_t p = 0; p < planes.size(); p++) {
    torch::Tensor f;
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    if (planes[p] == "u") {
      int64_t num_elements = x_filter_u_data.size();
      f = torch::from_blob(x_filter_u_data.data(), {num_elements}, options);
    }
    else if (planes[p] == "v") {
      int64_t num_elements = x_filter_v_data.size();
      f = torch::from_blob(x_filter_v_data.data(), {num_elements}, options);
    }
    else if (planes[p] == "y") {
      int64_t num_elements = x_filter_y_data.size();
      f = torch::from_blob(x_filter_y_data.data(), {num_elements}, options);
    }
    else {
      std::cout << "error!" << std::endl;
    }

    for (int i = 0; i < f.numel(); ++i) {
      size_t idx = idsmap[p][i];
      std::array<float, 1> input({f[i].item<float>()});
      (*filtcol)[idx] = FeatureVector<1>(input);
    }
  }
  e.put(std::move(filtcol), "filter");
}

DEFINE_ART_CLASS_TOOL(FilterDecoder)

#endif
