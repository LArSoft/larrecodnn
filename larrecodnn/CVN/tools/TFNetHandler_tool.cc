////////////////////////////////////////////////////////////////////////
/// \file    TFNetHandler.cxx
/// \brief   TFNetHandler for CVN
/// \author  Alexander Radovic - a.radovic@gmail.com
///          Leigh Whitehead   - leigh.howard.whitehead@cern.ch
///          Saul Alonso Monsalve - saul.alonso.monsalve@cern.ch
////////////////////////////////////////////////////////////////////////

#include "cetlib/getenv.h"
#include <iostream>
#include <string>

#include "art/Utilities/ToolMacros.h"
#include "canvas/Utilities/Exception.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "larrecodnn/CVN/func/CVNImageUtils.h"
#include "larrecodnn/CVN/interfaces/ITFNetHandler.h"
#include "larrecodnn/ImagePatternAlgs/Tensorflow/TF/tf_graph.h"

namespace lcvn {

  /// Wrapper for caffe::Net which handles construction and prediction
  class TFNetHandler : public ITFNetHandler {
  public:
    /// Constructor which takes a pset with DeployProto and ModelFile fields
    explicit TFNetHandler(const fhicl::ParameterSet& pset);

    /// Return prediction arrays for PixelMap
    std::vector<std::vector<float>> Predict(const PixelMap& pm) const override;

  private:
    std::string fLibPath; ///< Library path (typically dune_pardata...)
    std::string
      fTFProtoBuf; ///< location of the tf .pb file in the above path or the directory containing model files in SavedModel format (set UseBundle = true in this case)
    bool fUseLogChargeScale;         ///< Is the charge using a log scale?
    unsigned int fImageWires;        ///< Number of wires for the network to classify
    unsigned int fImageTDCs;         ///< Number of tdcs for the network to classify
    std::vector<bool> fReverseViews; ///< Do we need to reverse any views?
    bool fUseBundle; ///< Use a bundled model saved in the SavedModel format from Tensorflow
    std::unique_ptr<tf::Graph> fTFGraph; ///< Tensorflow graph
  };

  TFNetHandler::TFNetHandler(const fhicl::ParameterSet& pset)
    : fLibPath(cet::getenv(pset.get<std::string>("LibPath", "")))
    , fTFProtoBuf(fLibPath + "/" + pset.get<std::string>("TFProtoBuf"))
    , fUseLogChargeScale(pset.get<bool>("ChargeLogScale"))
    , fImageWires(pset.get<unsigned int>("NImageWires"))
    , fImageTDCs(pset.get<unsigned int>("NImageTDCs"))
    , fReverseViews(pset.get<std::vector<bool>>("ReverseViews"))
    , fUseBundle(pset.get<bool>("UseBundle"))
  {

    // Construct the TF Graph object. The empty vector {} is used since the protobuf
    // file gives the names of the output layer nodes
    mf::LogInfo("TFNetHandler") << "Loading network: " << fTFProtoBuf << std::endl;
    fTFGraph = tf::Graph::create(
      fTFProtoBuf.c_str(), {}, fUseBundle, pset.get<int>("NInputs"), pset.get<int>("NOutputs"));
    if (!fTFGraph) {
      art::Exception(art::errors::Unknown) << "Tensorflow model not found or incorrect";
    }
  }

  // Check the network outputs
  bool check(const std::vector<std::vector<float>>& outputs)
  {
    if (outputs.size() == 1) return true;
    size_t aux = 0;
    for (size_t o = 0; o < outputs.size(); ++o) {
      size_t aux2 = 0;

      for (size_t i = 0; i < outputs[o].size(); ++i)
        if (outputs[o][i] == 0.0 || outputs[o][i] == 1.0) aux2++;
      if (aux2 == outputs[o].size()) aux++;
    }
    return aux == outputs.size() ? false : true;
  }

  // Fill outputs with value -3
  void fillEmpty(std::vector<std::vector<float>>& outputs)
  {
    for (auto& output : outputs) {
      output.assign(output.size(), -3.0);
    }

    return;
  }

  std::vector<std::vector<float>> TFNetHandler::Predict(const PixelMap& pm) const
  {

    CVNImageUtils imageUtils(fImageWires, fImageTDCs, 3);
    // Configure the image utility
    imageUtils.SetViewReversal(fReverseViews);
    imageUtils.SetImageSize(fImageWires, fImageTDCs, 3);
    imageUtils.SetLogScale(fUseLogChargeScale);
    imageUtils.SetPixelMapSize(pm.NWire(), pm.NTdc());

    ImageVectorF thisImage;
    imageUtils.ConvertPixelMapToImageVectorF(pm, thisImage);
    std::vector<ImageVectorF> vecForTF;

    vecForTF.push_back(thisImage);

    std::vector<std::vector<std::vector<float>>>
      cvnResults; // shape(samples, #outputs, output_size)
    bool status = false;

    int counter = 0;

    while (status == false) { // do until it gets a correct result
      cvnResults = fTFGraph->run(vecForTF);
      status = check(cvnResults[0]);

      counter++;
      if (counter == 10) {
        std::cout << "Error, CVN never outputing a correct result. Filling result with zeros.";
        std::cout << std::endl;
        fillEmpty(cvnResults[0]);
        break;
      }
    };

    std::cout << "Classifier summary: ";
    std::cout << std::endl;
    int output_index = 0;
    for (auto const& output : cvnResults[0]) {
      std::cout << "Output " << output_index++ << ": ";
      for (auto const v : output)
        std::cout << v << ", ";
      std::cout << std::endl;
    }
    std::cout << std::endl;

    return cvnResults[0];
  }

}
DEFINE_ART_CLASS_TOOL(lcvn::TFNetHandler)
