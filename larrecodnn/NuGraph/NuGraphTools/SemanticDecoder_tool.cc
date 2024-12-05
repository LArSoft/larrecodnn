#ifndef SEMANTICDECODER_CXX
#define SEMANTICDECODER_CXX

#include "DecoderToolBase.h"

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include <torch/torch.h>

using anab::FeatureVector;
using anab::MVADescription;

class SemanticDecoder : public DecoderToolBase
{

public:

  /**
   *  @brief  Constructor
   *
   *  @param  pset
   */
  SemanticDecoder(const fhicl::ParameterSet &pset);

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
   */
  std::pair<DecoderType,std::string> declareProducts() override { return std::make_pair(Semantic5,"semantic"); };

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

SemanticDecoder::SemanticDecoder(const fhicl::ParameterSet &p)
{
  configure(p);
}

void SemanticDecoder::configure(const fhicl::ParameterSet& p) {
  //to do: take list of classes from config (and allow for variable number of classes)
}

void SemanticDecoder::writeEmptyToEvent(art::Event& e, const vector<vector<size_t> >& idsmap) {
  //
  std::unique_ptr<MVADescription<5>> semtdes(
                          new MVADescription<5>("nuslhits",//hitListHandle.provenance()->moduleLabel(), FIMXE
                          "semantic",
                          {"MIP", "HIP", "shower", "michel", "diffuse"}));
  e.put(std::move(semtdes), "semantic");
  //
  size_t size = 0;
  for (auto& v : idsmap) size += v.size();
  std::unique_ptr<vector<FeatureVector<5>>> semtcol(new vector<FeatureVector<5>>(size, FeatureVector<5>(std::array<float, 5>({-1., -1., -1., -1., -1.}))));
  e.put(std::move(semtcol), "semantic");
  //
}

void SemanticDecoder::writeToEvent(art::Event& e, const vector<vector<size_t> >& idsmap, const vector<NuGraphOutput>& infer_output) {
  //
  std::unique_ptr<MVADescription<5>> semtdes(
                          new MVADescription<5>("nuslhits",//hitListHandle.provenance()->moduleLabel(), FIMXE
                          "semantic",
                          {"MIP", "HIP", "shower", "michel", "diffuse"}));
  e.put(std::move(semtdes), "semantic");
  //
  size_t size = 0;
  for (auto& v : idsmap) size += v.size();
  std::unique_ptr<vector<FeatureVector<5>>> semtcol(new vector<FeatureVector<5>>(size, FeatureVector<5>(std::array<float, 5>({-1., -1., -1., -1., -1.}))));

  std::vector<float> x_semantic_u_data;
  std::vector<float> x_semantic_v_data;
  std::vector<float> x_semantic_y_data;

  for (auto& io : infer_output) {
    if (io.output_name == "x_semantic_u") x_semantic_u_data = io.output_vec;
    if (io.output_name == "x_semantic_v") x_semantic_v_data = io.output_vec;
    if (io.output_name == "x_semantic_y") x_semantic_y_data = io.output_vec;
  }

  if (debug) {
    std::cout << "x_semantic_u: " << std::endl;
    printVector(x_semantic_u_data);

    std::cout << "x_semantic_v: " << std::endl;
    printVector(x_semantic_v_data);

    std::cout << "x_semantic_y: " << std::endl;
    printVector(x_semantic_y_data);
  }

  size_t n_cols = 5;
  for (size_t p = 0; p < planes.size(); p++) {
    torch::Tensor s;
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    if (planes[p] == "u") {
      size_t n_rows = x_semantic_u_data.size() / n_cols;
      s = torch::from_blob(x_semantic_u_data.data(),
			   {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
			   options);
    }
    else if (planes[p] == "v") {
      size_t n_rows = x_semantic_v_data.size() / n_cols;
      s = torch::from_blob(x_semantic_v_data.data(),
			   {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
			   options);
    }
    else if (planes[p] == "y") {
      size_t n_rows = x_semantic_y_data.size() / n_cols;
      s = torch::from_blob(x_semantic_y_data.data(),
			   {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
			   options);
    }
    else {
      std::cout << "Error!!" << std::endl;
    }

    for (int i = 0; i < s.sizes()[0]; ++i) {
      size_t idx = idsmap[p][i];
      std::array<float, 5> input({s[i][0].item<float>(),
	    s[i][1].item<float>(),
	    s[i][2].item<float>(),
	    s[i][3].item<float>(),
	    s[i][4].item<float>()});
      softmax(input);
      FeatureVector<5> semt = FeatureVector<5>(input);
      (*semtcol)[idx] = semt;
    }
  }
  e.put(std::move(semtcol), "semantic");
}

DEFINE_ART_CLASS_TOOL(SemanticDecoder)

#endif
