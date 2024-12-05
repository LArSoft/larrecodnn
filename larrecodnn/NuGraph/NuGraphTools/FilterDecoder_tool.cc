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
  void configure(const fhicl::ParameterSet& p) {DecoderToolBase::configure(p);}

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

void FilterDecoder::writeEmptyToEvent(art::Event& e, const vector<vector<size_t> >& idsmap) {
  //
  size_t size = 0;
  for (auto& v : idsmap) size += v.size();
  std::unique_ptr<vector<FeatureVector<1>>> filtcol(new vector<FeatureVector<1>>(size, FeatureVector<1>(std::array<float, 1>({-1.}))));
  e.put(std::move(filtcol), instancename);
  //
}

void FilterDecoder::writeToEvent(art::Event& e, const vector<vector<size_t> >& idsmap, const vector<NuGraphOutput>& infer_output) {
  //
  size_t size = 0;
  for (auto& v : idsmap) size += v.size();
  std::unique_ptr<vector<FeatureVector<1>>> filtcol(new vector<FeatureVector<1>>(size, FeatureVector<1>(std::array<float, 1>({-1.}))));
  //
  for (size_t p = 0; p < planes.size(); p++) {
    //
    const std::vector<float>* x_filter_data = 0;
    for (auto& io : infer_output) {
      if (io.output_name == outputname+planes[p]) x_filter_data = &io.output_vec;
    }
    if (debug) {
      std::cout << outputname+planes[p] << std::endl;
      printVector(*x_filter_data);
    }
    //
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    int64_t num_elements = x_filter_data->size();
    const torch::Tensor f = torch::from_blob(const_cast<float*>(x_filter_data->data()), {num_elements}, options);
    //
    for (int i = 0; i < f.numel(); ++i) {
      size_t idx = idsmap[p][i];
      std::array<float, 1> input({f[i].item<float>()});
      (*filtcol)[idx] = FeatureVector<1>(input);
    }
  }
  e.put(std::move(filtcol), instancename);
}

DEFINE_ART_CLASS_TOOL(FilterDecoder)

#endif
