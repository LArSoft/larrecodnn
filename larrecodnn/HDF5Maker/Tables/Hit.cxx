#include "larrecodnn/HDF5Maker/Tables/Hit.h"

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
HitTable::HitTable(std::vector<Row> const& data)
  : Table("hits", HitColumns, data)
{}

//-----------------------------------------------------------------------------
void HitTable::Fill(art::Event const& evt)
{
  // get event ID
  art::EventID const& id = evt.id();

  // get service handle
  auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService>()->DataFor(evt);
  
  // loop over hits
  auto hits = evt.getHandle<std::vector<recob::Hit>>(fNuLabel);
  for (size_t i = 0; i < hits->size(); ++i) {
    recob::Hit const& hit = hits->at(i);

    geo::WireID wireid = hit.WireID();

    const geo::TPCGeo& tpc_geo = art::ServiceHandle<geo::Geometry>()->TPC(geo::TPCID{0,wireid.TPC});
    const geo::WireGeo& wire_geo = art::ServiceHandle<geo::WireReadout>()->Get().Wire(wireid);

    int plane = wireid.Plane;
    int wire = wireid.Wire;
    double time = clockData.TPCTick2Time(hit.PeakTime());

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
    double drift_distance = time * detProp.DriftVelocity();
    double drift = wire_center.X() - (drift_sign * drift_distance);

    fData.push_back({
      id.run(), id.subRun(), id.event(),  // event ID
      i, hit.Integral(), hit.RMS(),       // hit_id, integral, rms
      wireid.TPC, plane, wire, time,      // tpc, plane, wire, time
      view, proj, drift                   // view, proj, drift
    });

  } // for hit

} // function HitTable::Fill

} // namespace ng
