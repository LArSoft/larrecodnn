#include "larrecodnn/HDF5Maker/Tables/Hit.h"

#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "larcore/Geometry/Geometry.h"
#include "larcore/Geometry/WireReadout.h"
#include "larcorealg/Geometry/TPCGeo.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in hit table
std::vector<std::string> static const HitColumns
{
  "run", "subrun", "event", "hit_id", "integral", "rms", "tpc_id", "plane",
  "wire", "time", "view", "proj", "drift"
};

//-----------------------------------------------------------------------------
// hit table constructor
HitTable::HitTable(std::string const& hitLabel, std::vector<Row> const& data)
  : Table("hits", HitColumns, data), fHitLabel(hitLabel)
{}

//-----------------------------------------------------------------------------
void HitTable::Fill(art::Event const& evt)
{
  // get event ID
  art::EventID const& id = evt.id();

  // get service handle
  auto const dc = art::ServiceHandle<detinfo::DetectorClocksService>()->DataFor(evt);
  auto const dp = art::ServiceHandle<detinfo::DetectorPropertiesService>()->DataFor(evt, dc);
  
  // loop over hits
  auto hits = evt.getHandle<std::vector<recob::Hit>>(fHitLabel);
  for (size_t hit_id = 0; hit_id < hits->size(); ++hit_id) {
    recob::Hit const& hit = hits->at(hit_id);

    geo::WireID wireid = hit.WireID();

    const geo::TPCGeo& tpc_geo = art::ServiceHandle<geo::Geometry>()->TPC(geo::TPCID{0,wireid.TPC});
    const geo::WireGeo& wire_geo = art::ServiceHandle<geo::WireReadout>()->Get().Wire(wireid);

    int plane = wireid.Plane;
    int wire = wireid.Wire;
    double time = dc.TPCTick2Time(hit.PeakTime());

    // global view
    int view;
    if (plane == 2) {
      // view for collection plane is unchanged
      view = 2;
    } else {
      // separate induction hits out into views based on wire direction 
      view = static_cast<int>(wire_geo.CosThetaZ() < 0);
    } // if collection plane

    // global wire projection coordinate
    geo::Point_t wire_center = wire_geo.GetCenter();
    double proj = wire_center.Z() * wire_geo.SinThetaZ() - wire_center.Y() * wire_geo.CosThetaZ();

    // global drift time coordinate
    double drift_sign = geo::to_int(tpc_geo.DriftSign());
    double drift_distance = time * dp.DriftVelocity();
    double drift = wire_center.X() - (drift_sign * drift_distance);

    fData.push_back({
      id.run(), id.subRun(), id.event(),  // event ID
      hit_id, hit.Integral(), hit.RMS(),  // hit_id, integral, rms
      wireid.TPC, plane, wire, time,      // tpc, plane, wire, time
      view, proj, drift                   // view, proj, drift
    });

  } // for hit

} // function HitTable::Fill

} // namespace ng
