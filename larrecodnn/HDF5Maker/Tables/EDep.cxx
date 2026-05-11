#include "larrecodnn/HDF5Maker/Tables/EDep.h"

#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "larsim/MCCheater/BackTrackerService.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in energy deposit table
std::vector<std::string> static const EDepColumns
{
  "run", "subrun", "event", "ch_id", "g4_id", "energy",
  "position_x", "position_y", "position_z"
};

//-----------------------------------------------------------------------------
// edep table constructor
EDepTable::EDepTable(std::string const& chargeHitLabel,
                     std::vector<Row> const& data)
  : Table("edeps", EDepColumns, data), fChargeHitLabel(chargeHitLabel)
{}

//-----------------------------------------------------------------------------
void EDepTable::Fill(art::Event const& evt)
{
  // get event ID
  art::EventID const& id = evt.id();

  // get service handles
  art::ServiceHandle<cheat::BackTrackerService> bt;
  art::ServiceHandle<detinfo::DetectorClocksService> dc;

  // loop over hits
  auto const clock_data = dc->DataFor(evt);
  auto hits = evt.getHandle<std::vector<recob::Hit>>(fChargeHitLabel);
  for (size_t ch_id = 0; ch_id < hits->size(); ++ch_id) {
    recob::Hit const& hit = hits->at(ch_id);

    // skip events with no TrackIDEs
    if (!bt->HitToTrackIds(clock_data, hit).size()) continue;

    // loop over averaged sim::IDEs
    for (const sim::IDE& ide : bt->HitToAvgSimIDEs(clock_data, hit)) {

      // catch negative track IDs
      if (ide.trackID < 0) {
        throw art::Exception(art::errors::LogicError)
        << "Negative track ID (" << ide.trackID << ") found in simulated "
            "energy deposits! This is usually an indication that you're "
            "running over simulation from before the larsoft Geant4 "
            "refactor, which is not supported due to its incomplete MC "
            "truth record.";
      } // if track ID is negative

      fData.push_back({
        id.run(), id.subRun(), id.event(),  // event ID
        ch_id, ide.trackID, ide.energy,     // ch_id, g4_id, energy
        ide.x, ide.y, ide.z                 // position
      });

    } // for energy deposit
  } // for hit

} // function EDepTable::Fill

} // namespace ng
