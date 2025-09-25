#include "larrecodnn/HDF5Maker/Tables/EDep.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in edep table
std::vector<std::string> static const EDepColumns
{
  "run", "subrun", "event", "hit_id", "g4_id", "energy",
  "position_x", "position_y", "position_z"
};

//-----------------------------------------------------------------------------
// edep table constructor
EDepTable::EDepTable(std::vector<Row> const& data)
  : Table("edeps", EDepColumns, data)
{}

//-----------------------------------------------------------------------------
void EDepTable::Fill(art::Event const& evt)
{
  // get event ID
  art::EventID const& id = evt.id();

  // get service handles
  auto const bt = art::ServiceHandle<cheat::BackTrackerService>();
  auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService>()->DataFor(evt);

  // loop over hits
  auto hits = evt.getHandle<std::vector<recob::Hit>>(fNuLabel);
  for (size_t i = 0; i < hits->size(); ++i) {
    recob::Hit const& hit = hits->at(i);

    // skip events with no TrackIDEs
    if (!bt->HitToTrackIds(clockData, hit).size()) continue;

    // loop over averaged sim::IDEs
    for (const sim::IDE& ide : bt->HitToAvgSimIDEs(clockData, hit)) {
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
      hit.key(), ide.trackID, ide.energy, // hit_id, g4_id, energy
      ide.x, ide.y, ide.z                 // position
    });

    } // for energy deposit
  } // for hit

} // function EDepTable::Fill

} // namespace ng
