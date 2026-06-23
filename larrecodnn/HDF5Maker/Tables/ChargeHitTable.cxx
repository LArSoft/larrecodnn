#include "larrecodnn/HDF5Maker/Tables/ChargeHitTable.h"

#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "larcore/Geometry/Geometry.h"
#include "larcore/Geometry/WireReadout.h"
#include "larcorealg/Geometry/TPCGeo.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"
#include "lardataobj/RecoBase/Hit.h"

namespace nugraph {

  //-----------------------------------------------------------------------------
  // names of columns in charge hit table
  std::vector<std::string> static const ChargeHitColumns{"run",
                                                         "subrun",
                                                         "event",
                                                         "ch_id",
                                                         "integral",
                                                         "rms",
                                                         "tpc_id",
                                                         "plane",
                                                         "wire",
                                                         "time",
                                                         "view",
                                                         "proj",
                                                         "drift"};

  //-----------------------------------------------------------------------------
  // charge hit table constructor
  ChargeHitTable::ChargeHitTable(std::string const& chargeHitLabel, std::vector<Row> const& data)
    : Table("chargehits", ChargeHitColumns, data), fChargeHitLabel(chargeHitLabel)
  {}

  //-----------------------------------------------------------------------------
  void ChargeHitTable::Fill(art::Event const& evt)
  {
    // get event ID
    art::EventID const& id = evt.id();

    // get service handle
    art::ServiceHandle<detinfo::DetectorClocksService> dc;
    art::ServiceHandle<detinfo::DetectorPropertiesService> dp;

    // loop over hits
    auto const clock_data = dc->DataFor(evt);
    auto const det_prop = dp->DataFor(evt, clock_data);
    auto hits = evt.getHandle<std::vector<recob::Hit>>(fChargeHitLabel);
    for (size_t ch_id = 0; ch_id < hits->size(); ++ch_id) {
      recob::Hit const& hit = hits->at(ch_id);

      geo::WireID wireid = hit.WireID();

      const geo::TPCGeo& tpc_geo =
        art::ServiceHandle<geo::Geometry>()->TPC(geo::TPCID{0, wireid.TPC});
      const geo::WireGeo& wire_geo = art::ServiceHandle<geo::WireReadout>()->Get().Wire(wireid);

      int plane = wireid.Plane;
      int wire = wireid.Wire;
      double time = clock_data.TPCTick2Time(hit.PeakTime());

      // global view
      int view;
      if (plane == 2) {
        // view for collection plane is unchanged
        view = 2;
      }
      else {
        // separate induction hits out into views based on wire direction
        view = static_cast<int>(wire_geo.CosThetaZ() < 0);
      } // if collection plane

      // global wire projection coordinate
      geo::Point_t wire_center = wire_geo.GetCenter();
      double proj = wire_center.Z() * wire_geo.SinThetaZ() - wire_center.Y() * wire_geo.CosThetaZ();

      // global drift time coordinate
      double drift_sign = geo::to_int(tpc_geo.DriftSign());
      double drift_distance = time * det_prop.DriftVelocity();
      double drift = wire_center.X() - (drift_sign * drift_distance);

      fData.push_back({
        id.run(),
        id.subRun(),
        id.event(), // event ID
        ch_id,
        hit.Integral(),
        hit.RMS(), // ch_id, integral, rms
        wireid.TPC,
        plane,
        wire,
        time, // tpc, plane, wire, time
        view,
        proj,
        drift // view, proj, drift
      });

    } // for hit

  } // function ChargeHitTable::Fill

} // namespace nugraph
