#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Utilities/ToolMacros.h"
#include "canvas/Persistency/Common/FindManyP.h"

#include "larcore/Geometry/WireReadout.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/TrackHitMeta.h"
#include "lardataobj/RecoBase/Wire.h"
#include "nusimdata/SimulationBase/MCTruth.h"

#include "ImageMaker.h"
#include "TTimeStamp.h"
#include "hep_hpc/hdf5/Ntuple.hpp"
#include "hep_hpc/hdf5/make_ntuple.hpp"

using namespace hep_hpc::hdf5;

namespace dnn {

  class SavePiMu : public ImageMaker {
  public:
    explicit SavePiMu(fhicl::ParameterSet const& ps);

    void saveImage(art::Event const& e, hep_hpc::hdf5::File& hdffile) override;

  private:
    art::InputTag fTrackModuleLabel;
    art::InputTag fWireModuleLabel;
    art::InputTag fMCTruthLabel;
  };

  SavePiMu::SavePiMu(fhicl::ParameterSet const& ps)
    : fTrackModuleLabel{ps.get<art::InputTag>("TrackModuleLabel")}
    , fWireModuleLabel{ps.get<art::InputTag>("WireModuleLabel")}
    , fMCTruthLabel{ps.get<art::InputTag>("MCTruthLabel")}
  {}

  void SavePiMu::saveImage(art::Event const& e, hep_hpc::hdf5::File& hdffile)
  {
    auto const& wireReadoutGeom = art::ServiceHandle<geo::WireReadout>()->Get();

    static Ntuple<unsigned int, unsigned int, unsigned int, double> evtids(
      hdffile, "evtids", {"run", "subrun", "event", "evttime"});

    static auto image =
      make_ntuple({hdffile, "image", 1000},
                  make_scalar_column<unsigned short>("label"),
                  make_column<float, 2>(
                    "adc",                                // 2 means each element is a 2-d array
                    {50, 50},                             // extent of each array dimension
                    1024 * 1024 / (2500 * sizeof(float)), // chunk size
                    {PropertyList{H5P_DATASET_CREATE}(&H5Pset_shuffle)(&H5Pset_deflate, 6u)}));

    // Get event information
    double evttime;

    art::Timestamp ts = e.time();
    if (ts.timeHigh() == 0) {
      TTimeStamp tts(ts.timeLow());
      evttime = tts.AsDouble();
    }
    else {
      TTimeStamp tts(ts.timeHigh(), ts.timeLow());
      evttime = tts.AsDouble();
    }

    // Get image information
    // * tracks
    auto const& tracks = e.getProduct<std::vector<recob::Track>>(fTrackModuleLabel);
    auto trackListHandle = e.getHandle<std::vector<recob::Track>>(fTrackModuleLabel);
    art::FindManyP<recob::Hit, recob::TrackHitMeta> fmthm(trackListHandle, e, fTrackModuleLabel);

    // * wires
    auto const& wires = e.getProduct<std::vector<recob::Wire>>(fWireModuleLabel);

    // * MC truth
    auto const& mclist = e.getProduct<std::vector<simb::MCTruth>>(fMCTruthLabel);

    unsigned short flag = 999;
    if (!mclist.empty()) {
      auto particle = mclist[0].GetParticle(0);
      if (std::abs(particle.PdgCode()) == 13) flag = 0;
      if (std::abs(particle.PdgCode()) == 211) flag = 1;
    }

    if (!tracks.empty()) {
      auto trkend = tracks[0].End();
      if (trkend.X() > -360 + 50 && trkend.X() < 360 - 50 && trkend.Y() > 50 &&
          trkend.Y() < 610 - 50 && trkend.Z() > 50 && trkend.Z() < 710 - 50) {
        if (fmthm.isValid()) {
          auto vhit = fmthm.at(0);
          auto vmeta = fmthm.data(0);
          int ihit = -1;
          int maxindex = -1;
          for (size_t i = 0; i < vhit.size(); ++i) {
            if (static_cast<int>(vmeta[i]->Index()) == std::numeric_limits<int>::max()) {
              continue;
            }
            if (vhit[i]->WireID().Plane == 2) {
              if (int(vmeta[i]->Index()) > maxindex) {
                maxindex = vmeta[i]->Index();
                ihit = i;
              }
            }
          }
          if (ihit >= 0) {
            auto endwire = vhit[ihit]->WireID();
            float endtime = vhit[ihit]->PeakTime();
            float adc[50][50] = {{0.}};
            for (auto& wire : wires) {
              int channel = wire.Channel();
              auto wireids = wireReadoutGeom.ChannelToWire(channel);
              if (wireids[0].Plane == 2 && wireids[0].TPC == endwire.TPC &&
                  wireids[0].Wire >= endwire.Wire - 25 && wireids[0].Wire < endwire.Wire + 25) {
                int idx = wireids[0].Wire - endwire.Wire + 25;
                const recob::Wire::RegionsOfInterest_t& signalROI = wire.SignalROI();
                int lasttick = 0;
                for (const auto& range : signalROI.get_ranges()) {
                  const auto& waveform = range.data();
                  // ROI start time
                  raw::TDCtick_t roiFirstBinTick = range.begin_index();
                  for (int i = lasttick; i < roiFirstBinTick; ++i) {
                    if (i >= int(endtime) - 25 && i < int(endtime) + 25) {
                      adc[idx][i - int(endtime) - 25] = 0;
                    }
                  }
                  lasttick = roiFirstBinTick;
                  for (size_t i = 0; i < waveform.size(); ++i) {
                    if (lasttick >= int(endtime) - 25 && lasttick < int(endtime) + 25) {
                      adc[idx][lasttick - int(endtime) + 25] = waveform[i];
                      ++lasttick;
                    }
                  }
                }
                for (int i = lasttick; i < 6000; ++i) {
                  if (i >= int(endtime) - 25 && i < int(endtime) + 25) {
                    adc[idx][i - int(endtime) + 25] = 0;
                  }
                }
              }
            } // Loop over all wire signals
            evtids.insert(e.run(), e.subRun(), e.id().event(), evttime);
            image.insert(flag, &adc[0][0]);
          }
        }
      }
    }

    return;
  }
}

DEFINE_ART_CLASS_TOOL(dnn::SavePiMu)
