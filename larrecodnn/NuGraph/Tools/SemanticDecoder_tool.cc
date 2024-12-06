#ifndef SEMANTICDECODER_CXX
#define SEMANTICDECODER_CXX

#include "DecoderToolBase.h"

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include <torch/torch.h>

using anab::FeatureVector;
using anab::MVADescription;

// fixme: this only works for 5 categories and should be extended to different sizes. This may require making the class templated.
class SemanticDecoder : public DecoderToolBase {

public:
  /**
   *  @brief  Constructor
   *
   *  @param  pset
   */
  SemanticDecoder(const fhicl::ParameterSet& pset);

  /**
   *  @brief  Virtual Destructor
   */
  virtual ~SemanticDecoder() noexcept = default;

  /**
   *  @brief Interface for configuring the particular algorithm tool
   *
   *  @param ParameterSet  The input set of parameters for configuration
   */
  void configure(const fhicl::ParameterSet&);

  /**
   * @brief declareProducts function
   *
   * @param art::ProducesCollector
   */
  void declareProducts(art::ProducesCollector& collector) override
  {
    collector.produces<vector<FeatureVector<5>>>(instancename);
    collector.produces<MVADescription<5>>(instancename);
  }

  /**
   * @brief writeEmptyToEvent function
   *
   * @param art::Event event record
   */
  void writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap) override;

  /**
   * @brief Decoder function
   *
   * @param art::Event event record for decoder
   */
  void writeToEvent(art::Event& e,
                    const vector<vector<size_t>>& idsmap,
                    const vector<NuGraphOutput>& infer_output) override;

private:
  std::vector<std::string> categories;
  art::InputTag hitInput;
};

SemanticDecoder::SemanticDecoder(const fhicl::ParameterSet& p)
{
  configure(p);
}

void SemanticDecoder::configure(const fhicl::ParameterSet& p)
{
  DecoderToolBase::configure(p);
  categories = p.get<std::vector<std::string>>("categories");
  hitInput = p.get<art::InputTag>("hitInput");
}

void SemanticDecoder::writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap)
{
  //
  std::unique_ptr<MVADescription<5>> semtdes(
    new MVADescription<5>(hitInput.label(), instancename, categories));
  e.put(std::move(semtdes), instancename);
  //
  size_t size = 0;
  for (auto& v : idsmap)
    size += v.size();
  std::array<float, 5> arr;
  std::fill(arr.begin(), arr.end(), -1.);
  std::unique_ptr<vector<FeatureVector<5>>> semtcol(
    new vector<FeatureVector<5>>(size, FeatureVector<5>(arr)));
  e.put(std::move(semtcol), instancename);
  //
}

void SemanticDecoder::writeToEvent(art::Event& e,
                                   const vector<vector<size_t>>& idsmap,
                                   const vector<NuGraphOutput>& infer_output)
{
  //
  std::unique_ptr<MVADescription<5>> semtdes(
    new MVADescription<5>(hitInput.label(), instancename, categories));
  e.put(std::move(semtdes), instancename);
  //
  size_t size = 0;
  for (auto& v : idsmap)
    size += v.size();
  std::array<float, 5> arr;
  std::fill(arr.begin(), arr.end(), -1.);
  std::unique_ptr<vector<FeatureVector<5>>> semtcol(
    new vector<FeatureVector<5>>(size, FeatureVector<5>(arr)));

  size_t n_cols = categories.size();
  for (size_t p = 0; p < planes.size(); p++) {
    //
    const std::vector<float>* x_semantic_data = 0;
    for (auto& io : infer_output) {
      if (io.output_name == outputname + planes[p]) x_semantic_data = &io.output_vec;
    }
    if (debug) {
      std::cout << outputname + planes[p] << std::endl;
      printVector(*x_semantic_data);
    }

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    size_t n_rows = x_semantic_data->size() / n_cols;
    const torch::Tensor s =
      torch::from_blob(const_cast<float*>(x_semantic_data->data()),
                       {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
                       options);

    for (int i = 0; i < s.sizes()[0]; ++i) {
      size_t idx = idsmap[p][i];
      std::array<float, 5> input;
      for (size_t j = 0; j < n_cols; ++j)
        input[j] = s[i][j].item<float>();
      softmax(input);
      FeatureVector<5> semt = FeatureVector<5>(input);
      (*semtcol)[idx] = semt;
    }
  }
  e.put(std::move(semtcol), instancename);
}

DEFINE_ART_CLASS_TOOL(SemanticDecoder)

#endif
