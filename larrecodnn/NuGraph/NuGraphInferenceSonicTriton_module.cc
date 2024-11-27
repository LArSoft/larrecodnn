////////////////////////////////////////////////////////////////////////
// Class:       NuGraphInferenceSonicTriton
// Plugin Type: producer (Unknown Unknown)
// File:        NuGraphInferenceSonicTriton_module.cc
//
// Generated at Tue Nov 14 14:41:30 2023 by Giuseppe Cerati using cetskelgen
// from  version .
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Utilities/ToolMacros.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "fhiclcpp/types/Table.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include <array>
#include <limits>
#include <memory>

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "lardataobj/RecoBase/Vertex.h" //this creates a conflict with torch script if included before it...
#include "larrecodnn/ImagePatternAlgs/NuSonic/Triton/TritonClient.h"
#include "larrecodnn/ImagePatternAlgs/NuSonic/Triton/TritonData.h"

#include <getopt.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

class NuGraphInferenceSonicTriton;

using anab::FeatureVector;
using anab::MVADescription;
using recob::Hit;
using recob::SpacePoint;
using std::array;
using std::vector;

namespace {

  template <typename T, size_t N>
  void softmax(std::array<T, N>& arr)
  {
    T m = -std::numeric_limits<T>::max();
    for (size_t i = 0; i < arr.size(); i++) {
      if (arr[i] > m) { m = arr[i]; }
    }
    T sum = 0.0;
    for (size_t i = 0; i < arr.size(); i++) {
      sum += expf(arr[i] - m);
    }
    T offset = m + logf(sum);
    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = expf(arr[i] - offset);
    }
    return;
  }
}

// Function to print elements of a vector<float>
void printVector(const std::vector<float>& vec)
{
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << vec[i];
    // Print space unless it's the last element
    if (i != vec.size() - 1) { std::cout << " "; }
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

class NuGraphInferenceSonicTriton : public art::EDProducer {
public:
  explicit NuGraphInferenceSonicTriton(fhicl::ParameterSet const& p);

  // Plugins should not be copied or assigned.
  NuGraphInferenceSonicTriton(NuGraphInferenceSonicTriton const&) = delete;
  NuGraphInferenceSonicTriton(NuGraphInferenceSonicTriton&&) = delete;
  NuGraphInferenceSonicTriton& operator=(NuGraphInferenceSonicTriton const&) = delete;
  NuGraphInferenceSonicTriton& operator=(NuGraphInferenceSonicTriton&&) = delete;

  // Required functions.
  void produce(art::Event& e) override;

private:
  vector<std::string> planes;
  art::InputTag hitInput;
  art::InputTag spsInput;
  size_t minHits;
  bool debug;
  bool filterDecoder;
  bool semanticDecoder;
  bool vertexDecoder;
  fhicl::ParameterSet tritonPset;
  std::unique_ptr<lartriton::TritonClient> triton_client;

  template<class T> void setShapeAndToServer(lartriton::TritonData<triton::client::InferInput>& triton_input, vector<T>& vec, size_t batchSize) {
    triton_input.setShape({static_cast<long int>(vec.size())});
    triton_input.toServer( std::make_shared<lartriton::TritonInput<T>>(lartriton::TritonInput<T>(batchSize,vec))  );
  }

};

NuGraphInferenceSonicTriton::NuGraphInferenceSonicTriton(fhicl::ParameterSet const& p)
  : EDProducer{p}
  , planes(p.get<vector<std::string>>("planes"))
  , hitInput(p.get<art::InputTag>("hitInput"))
  , spsInput(p.get<art::InputTag>("spsInput"))
  , minHits(p.get<size_t>("minHits"))
  , debug(p.get<bool>("debug"))
  , filterDecoder(p.get<bool>("filterDecoder"))
  , semanticDecoder(p.get<bool>("semanticDecoder"))
  , vertexDecoder(p.get<bool>("vertexDecoder"))
  , tritonPset(p.get<fhicl::ParameterSet>("TritonConfig"))
{

  // ... Create the Triton inference client
  if (debug) std::cout << "TritonConfig: " << tritonPset.to_string() << std::endl;
  triton_client = std::make_unique<lartriton::TritonClient>(tritonPset);

  if (filterDecoder) { produces<vector<FeatureVector<1>>>("filter"); }
  //
  if (semanticDecoder) {
    produces<vector<FeatureVector<5>>>("semantic");
    produces<MVADescription<5>>("semantic");
  }
  //
  if (vertexDecoder) { produces<vector<recob::Vertex>>("vertex"); }
}

void NuGraphInferenceSonicTriton::produce(art::Event& e)
{

  // Graph inputs
  vector<int32_t> hit_table_hit_id_data;
  vector<int32_t> hit_table_local_plane_data;
  vector<float> hit_table_local_time_data;
  vector<int32_t> hit_table_local_wire_data;
  vector<float> hit_table_integral_data;
  vector<float> hit_table_rms_data;
  vector<int32_t> spacepoint_table_spacepoint_id_data;
  vector<int32_t> spacepoint_table_hit_id_u_data;
  vector<int32_t> spacepoint_table_hit_id_v_data;
  vector<int32_t> spacepoint_table_hit_id_y_data;

  //////
  art::Handle<vector<Hit>> hitListHandle;
  vector<art::Ptr<Hit>> hitlist;
  if (e.getByLabel(hitInput, hitListHandle)) { art::fill_ptr_vector(hitlist, hitListHandle); }

  vector<vector<size_t>> idsmap(planes.size(), vector<size_t>());
  for (auto h : hitlist) {
    idsmap[h->View()].push_back(h.key());
  }

  // hit table
  for (auto h : hitlist) {
    hit_table_hit_id_data.push_back(h.key());
    hit_table_local_plane_data.push_back(h->View());
    hit_table_local_time_data.push_back(h->PeakTime());
    hit_table_local_wire_data.push_back(h->WireID().Wire);
    hit_table_integral_data.push_back(h->Integral());
    hit_table_rms_data.push_back(h->RMS());
  }

  // Get spacepoints from the event record
  art::Handle<vector<SpacePoint>> spListHandle;
  vector<art::Ptr<SpacePoint>> splist;
  if (e.getByLabel(spsInput, spListHandle)) { art::fill_ptr_vector(splist, spListHandle); }
  // Get assocations from spacepoints to hits
  vector<vector<art::Ptr<Hit>>> sp2Hit(splist.size());
  if (splist.size() > 0) {
    art::FindManyP<Hit> fmp(spListHandle, e, "sps");
    for (size_t spIdx = 0; spIdx < sp2Hit.size(); ++spIdx) {
      sp2Hit[spIdx] = fmp.at(spIdx);
    }
  }

  // space point table
  for (size_t i = 0; i < splist.size(); ++i) {
    spacepoint_table_spacepoint_id_data.push_back(i);
    spacepoint_table_hit_id_u_data.push_back(-1);
    spacepoint_table_hit_id_v_data.push_back(-1);
    spacepoint_table_hit_id_y_data.push_back(-1);
    for (size_t j = 0; j < sp2Hit[i].size(); ++j) {
      if (sp2Hit[i][j]->View() == 0) spacepoint_table_hit_id_u_data.back() = sp2Hit[i][j].key();
      if (sp2Hit[i][j]->View() == 1) spacepoint_table_hit_id_v_data.back() = sp2Hit[i][j].key();
      if (sp2Hit[i][j]->View() == 2) spacepoint_table_hit_id_y_data.back() = sp2Hit[i][j].key();
    }
  }
  ///

  std::unique_ptr<vector<FeatureVector<1>>> filtcol(
    new vector<FeatureVector<1>>(hitlist.size(), FeatureVector<1>(std::array<float, 1>({-1.}))));

  std::unique_ptr<vector<FeatureVector<5>>> semtcol(new vector<FeatureVector<5>>(
    hitlist.size(), FeatureVector<5>(std::array<float, 5>({-1., -1., -1., -1., -1.}))));
  std::unique_ptr<MVADescription<5>> semtdes(
    new MVADescription<5>(hitListHandle.provenance()->moduleLabel(),
                          "semantic",
                          {"MIP", "HIP", "shower", "michel", "diffuse"}));

  std::unique_ptr<vector<recob::Vertex>> vertcol(new vector<recob::Vertex>());

  if (debug) std::cout << "Hits size=" << hitlist.size() << std::endl;
  if (hitlist.size() < minHits) {
    if (filterDecoder) { e.put(std::move(filtcol), "filter"); }
    if (semanticDecoder) {
      e.put(std::move(semtcol), "semantic");
      e.put(std::move(semtdes), "semantic");
    }
    if (vertexDecoder) { e.put(std::move(vertcol), "vertex"); }
    return;
  }

  // NuSonic Triton Server

  auto start = std::chrono::high_resolution_clock::now();

  //Here the input should be sent to Triton
  triton_client->reset();
  size_t batchSize = 1;//the code below assumes/has only been tested for batch size = 1
  triton_client->setBatchSize(batchSize); // set batch size

  auto& inputs = triton_client->input();
  for (auto& input_pair : inputs) {
    const std::string& key = input_pair.first;
    auto& triton_input = input_pair.second;

    if (key == "hit_table_hit_id") {
      setShapeAndToServer(triton_input,hit_table_hit_id_data,batchSize);
    }
    else if (key == "hit_table_local_plane") {
      setShapeAndToServer(triton_input,hit_table_local_plane_data,batchSize);
    }
    else if (key == "hit_table_local_time") {
      setShapeAndToServer(triton_input,hit_table_local_time_data,batchSize);
    }
    else if (key == "hit_table_local_wire") {
      setShapeAndToServer(triton_input,hit_table_local_wire_data,batchSize);
    }
    else if (key == "hit_table_integral") {
      setShapeAndToServer(triton_input,hit_table_integral_data,batchSize);
    }
    else if (key == "hit_table_rms") {
      setShapeAndToServer(triton_input,hit_table_rms_data,batchSize);
    }
    else if (key == "spacepoint_table_spacepoint_id") {
      setShapeAndToServer(triton_input,spacepoint_table_spacepoint_id_data,batchSize);
    }
    else if (key == "spacepoint_table_hit_id_u") {
      setShapeAndToServer(triton_input,spacepoint_table_hit_id_u_data,batchSize);
    }
    else if (key == "spacepoint_table_hit_id_v") {
      setShapeAndToServer(triton_input,spacepoint_table_hit_id_v_data,batchSize);
    }
    else if (key == "spacepoint_table_hit_id_y") {
      setShapeAndToServer(triton_input,spacepoint_table_hit_id_y_data,batchSize);
    }
    else {
      throw std::runtime_error(std::string("Error -- key " + key + " not supported!"));
    }
  }

  // ~~~~ Send inference request
  triton_client->dispatch();
  // ~~~~ Retrieve inference results
  auto& infer_result = triton_client->output();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken for inference: " << elapsed.count() << " seconds" << std::endl;

  // Writing the outputs to the output root file

  if (semanticDecoder) {

    const auto& triton_output0 = infer_result.at("x_semantic_u");
    const auto& prob0 = triton_output0.fromServer<float>();
    size_t triton_input0_elements = std::distance(prob0[0].begin(), prob0[0].end());

    const auto& triton_output1 = infer_result.at("x_semantic_v");
    const auto& prob1 = triton_output1.fromServer<float>();
    size_t triton_input1_elements = std::distance(prob1[0].begin(), prob1[0].end());

    const auto& triton_output2 = infer_result.at("x_semantic_y");
    const auto& prob2 = triton_output2.fromServer<float>();
    size_t triton_input2_elements = std::distance(prob2[0].begin(), prob2[0].end());

    std::vector<float> x_semantic_u_data;
    x_semantic_u_data.reserve(triton_input0_elements);
    x_semantic_u_data.insert(x_semantic_u_data.end(), prob0[0].begin(), prob0[0].end());

    std::vector<float> x_semantic_v_data;
    x_semantic_v_data.reserve(triton_input1_elements);
    x_semantic_v_data.insert(x_semantic_v_data.end(), prob1[0].begin(), prob1[0].end());

    std::vector<float> x_semantic_y_data;
    x_semantic_y_data.reserve(triton_input2_elements);
    x_semantic_y_data.insert(x_semantic_y_data.end(), prob2[0].begin(), prob2[0].end());

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
    e.put(std::move(semtdes), "semantic");
  }

  if (filterDecoder) {

    const auto& triton_output3 = infer_result.at("x_filter_u");
    const auto& prob3 = triton_output3.fromServer<float>();
    size_t triton_input3_elements = std::distance(prob3[0].begin(), prob3[0].end());

    const auto& triton_output4 = infer_result.at("x_filter_v");
    const auto& prob4 = triton_output4.fromServer<float>();
    size_t triton_input4_elements = std::distance(prob4[0].begin(), prob4[0].end());

    const auto& triton_output5 = infer_result.at("x_filter_y");
    const auto& prob5 = triton_output5.fromServer<float>();
    size_t triton_input5_elements = std::distance(prob5[0].begin(), prob5[0].end());

    std::vector<float> x_filter_u_data;
    x_filter_u_data.reserve(triton_input3_elements);
    x_filter_u_data.insert(x_filter_u_data.end(), prob3[0].begin(), prob3[0].end());

    std::vector<float> x_filter_v_data;
    x_filter_v_data.reserve(triton_input4_elements);
    x_filter_v_data.insert(x_filter_v_data.end(), prob4[0].begin(), prob4[0].end());

    std::vector<float> x_filter_y_data;
    x_filter_y_data.reserve(triton_input5_elements);
    x_filter_y_data.insert(x_filter_y_data.end(), prob5[0].begin(), prob5[0].end());

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

  if (vertexDecoder) { e.put(std::move(vertcol), "vertex"); }
}

DEFINE_ART_MODULE(NuGraphInferenceSonicTriton)
