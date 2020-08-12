////////////////////////////////////////////////////////////////////////////////////////////////////
// Class:       IPointIdAlg (interface for tool version of PointIdAlg)
// Authors:     D.Stefan (Dorota.Stefan@ncbj.gov.pl),         from DUNE, CERN/NCBJ, since May 2016
//              R.Sulej (Robert.Sulej@cern.ch),               from DUNE, FNAL/NCBJ, since May 2016
//              P.Plonski,                                    from DUNE, WUT,       since May 2016
//              M.Wang,                                       from DUNE, FNAL, 2020: tool version
//
//
// Point Identification Algorithm
//
//      Run CNN or MLP trained to classify a point in 2D projection. Various features can be
//      recognized, depending on the net model/weights used.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef IPointIdAlg_H
#define IPointIdAlg_H

#include "fhiclcpp/types/OptionalAtom.h"
#include "fhiclcpp/types/OptionalSequence.h"
#include "larreco/RecoAlg/ImagePatternAlgs/DataProvider/DataProviderAlg.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

namespace PointIdAlgTools {
  class IPointIdAlg : virtual public img::DataProviderAlg {
  public:
    struct Config : public img::DataProviderAlg::Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::OptionalAtom<std::string> NNetModelFile{Name("NNetModelFile"),
                                                     Comment("Neural net model to apply.")};
      fhicl::Sequence<std::string> NNetOutputs{Name("NNetOutputs"),
                                               Comment("Labels of the network outputs.")};
      fhicl::OptionalSequence<std::string> NNetOutputPattern{
        Name("NNetOutputPattern"),
        Comment("Pattern to use when searching for network outputs.")};
      fhicl::Atom<unsigned int> PatchSizeW{Name("PatchSizeW"), Comment("How many wires in patch.")};
      fhicl::Atom<unsigned int> PatchSizeD{Name("PatchSizeD"),
                                           Comment("How many downsampled ADC entries in patch")};
      fhicl::OptionalAtom<std::string> ToolType{Name("tool_type"),
                                                Comment("PointID algorithm tool type")};
      fhicl::OptionalAtom<std::string> TrtisModelName{
        Name("TrtisModelName"),
        Comment("Model directory name in repository of TensorRT inference server")};
      fhicl::OptionalAtom<std::string> TrtisURL{Name("TrtisURL"),
                                                Comment("URL of TensorRT inference server")};
      fhicl::OptionalAtom<int64_t> TrtisModelVersion{
        Name("TrtisModelVersion"),
        Comment("Version number of TensorRT inference server model")};
      fhicl::OptionalAtom<bool> TrtisVerbose{
        Name("TrtisVerbose"),
        Comment("Verbosity switch for TensorRT inference server client")};
    };
    virtual ~IPointIdAlg() noexcept = default;

    // Define standard art tool interface
    //virtual void configure(const fhicl::ParameterSet& pset) = 0;

    virtual std::vector<float> Run(std::vector<std::vector<float>> const& inp2d) const = 0;
    virtual std::vector<std::vector<float>> Run(
      std::vector<std::vector<std::vector<float>>> const& inps,
      int samples = -1) const = 0;

    // calculate single-value prediction (2-class probability) for [wire, drift] point
    float
    predictIdValue(unsigned int wire, float drift, size_t outIdx = 0)
    {
      float result = 0.;

      if (!bufferPatch(wire, drift)) {
        mf::LogError("PointIdAlg") << "Patch buffering failed.";
        return result;
      }

      auto out = Run(fWireDriftPatch);
      if (!out.empty()) { result = out[outIdx]; }
      else {
        mf::LogError("PointIdAlg") << "Problem with applying model to input.";
      }

      return result;
    }

    // Calculate multi-class probabilities for [wire, drift] point
    std::vector<float>
    predictIdVector(unsigned int wire, float drift)
    {
      std::vector<float> result;

      if (!bufferPatch(wire, drift)) {
        mf::LogError("PointIdAlg") << "Patch buffering failed.";
        return result;
      }

      result = Run(fWireDriftPatch);
      if (result.empty()) { mf::LogError("PointIdAlg") << "Problem with applying model to input."; }

      return result;
    }

    // Calculate multi-class probabilities for a vector of [wire, drift] points
    std::vector<std::vector<float>>
    predictIdVectors(const std::vector<std::pair<unsigned int, float>>& points)
    {
      if (points.empty()) { return std::vector<std::vector<float>>(); }

      std::vector<std::vector<std::vector<float>>> inps(
        points.size(),
        std::vector<std::vector<float>>(fPatchSizeW, std::vector<float>(fPatchSizeD)));
      for (size_t i = 0; i < points.size(); ++i) {
        unsigned int wire = points[i].first;
        float drift = points[i].second;
        if (!bufferPatch(wire, drift, inps[i])) {
          throw cet::exception("PointIdAlg") << "Patch buffering failed" << std::endl;
        }
      }

      return Run(inps);
    }

    std::vector<std::string> const&
    outputLabels(void) const
    {
      return fNNetOutputs;
    }
    bool
    isInsideFiducialRegion(unsigned int wire, float drift) const
    {
      size_t marginW = fPatchSizeW / 8; // fPatchSizeX/2 will make patch always completely filled
      size_t marginD = fPatchSizeD / 8;

      size_t scaledDrift = (size_t)(drift / fDriftWindow);
      if ((wire >= marginW) && (wire < fNWires - marginW) && (scaledDrift >= marginD) &&
          (scaledDrift < fNScaledDrifts - marginD)) {
        return true;
      }
      else {
        return false;
      }
    }

  protected:
    std::vector<std::string> fNNetOutputs;
    size_t fPatchSizeW, fPatchSizeD;
    std::vector<std::vector<float>> fWireDriftPatch; // patch data around the identified point
    size_t fCurrentWireIdx, fCurrentScaledDrift;

    bool
    bufferPatch(size_t wire, float drift, std::vector<std::vector<float>>& patch)
    {
      if (fDownscaleFullView) {
        size_t sd = (size_t)(drift / fDriftWindow);
        if ((fCurrentWireIdx == wire) && (fCurrentScaledDrift == sd))
          return true; // still within the current position

        fCurrentWireIdx = wire;
        fCurrentScaledDrift = sd;

        return patchFromDownsampledView(wire, drift, fPatchSizeW, fPatchSizeD, patch);
      }
      else {
        if ((fCurrentWireIdx == wire) && (fCurrentScaledDrift == drift))
          return true; // still within the current position

        fCurrentWireIdx = wire;
        fCurrentScaledDrift = drift;

        return patchFromOriginalView(wire, drift, fPatchSizeW, fPatchSizeD, patch);
      }
    }

    bool
    bufferPatch(size_t wire, float drift)
    {
      return bufferPatch(wire, drift, fWireDriftPatch);
    }

    void
    resizePatch(void)
    {
      fWireDriftPatch.resize(fPatchSizeW);
      for (auto& r : fWireDriftPatch)
        r.resize(fPatchSizeD);
    }
  };
}

#endif
