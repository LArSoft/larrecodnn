#include "larrecodnn/HDF5Maker/Tables/Particle.h"

#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "larsim/MCCheater/BackTrackerService.h"
#include "larsim/MCCheater/ParticleInventoryService.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in particle table
std::vector<std::string> static const ParticleColumns
{
  "run", "subrun", "event", "g4_id", "type", "parent_id",
  "mom_x", "mom_y", "mom_z", "mom_e",
  "start_x", "start_y", "start_z", "start_t",
  "end_x", "end_y", "end_z", "end_t",
  "start_process", "end_process"
};

//-----------------------------------------------------------------------------
// particle table constructor
ParticleTable::ParticleTable(std::string const& hitLabel,
                             std::vector<Row> const& data)
  : Table("particles", ParticleColumns, data), fHitLabel(hitLabel)
{}

//-----------------------------------------------------------------------------
void ParticleTable::Fill(art::Event const& evt)
{
  // get event ID
  art::EventID const& id = evt.id();

  // get service handles
  art::ServiceHandle<detinfo::DetectorClocksService> dc;
  art::ServiceHandle<cheat::BackTrackerService> bt;
  art::ServiceHandle<cheat::ParticleInventoryService> pi;

  // use energy deposits to get visible particles
  std::set<int> visible_ids;
  auto const clock_data = dc->DataFor(evt);
  auto hits = evt.getHandle<std::vector<recob::Hit>>(fHitLabel);
  for (size_t hit_id = 0; hit_id < hits->size(); ++hit_id) {
    recob::Hit const& hit = hits->at(hit_id);

    // skip events with no TrackIDEs
    if (!bt->HitToTrackIds(clock_data, hit).size()) continue;

    // add visible particle IDs
    for (const sim::IDE& ide : bt->HitToAvgSimIDEs(clock_data, hit)) {
      visible_ids.insert(abs(ide.trackID));
    } // for energy deposit
  } // for hit

  // add invisible particles with visible children to ensure unbroken hierarchy
  std::set<int> g4_ids = visible_ids;
  for (int id : visible_ids) {
    simb::MCParticle const* p = pi->TrackIdToParticle_P(abs(id));
    while (p->Mother() != 0) {
      g4_ids.insert(abs(p->Mother()));
      p = pi->TrackIdToParticle_P(abs(p->Mother()));
    } // while particle is not primary
  } // for visible particle

  // Loop over true particles and fill table
  for (int g4_id : g4_ids) {
    simb::MCParticle const& p = pi->TrackIdToParticle(g4_id);
    fData.push_back({
      id.run(), id.subRun(), id.event(),      // event ID, g4_id
      g4_id, p.PdgCode(), p.Mother(),         // g4_id, type, parent_id
      p.Px(), p.Py(), p.Pz(), p.E(),          // momentum
      p.Vx(), p.Vy(), p.Vz(), p.T(),          // start position
      p.EndX(), p.EndY(), p.EndZ(), p.EndT(), // end position
      p.Process(), p.EndProcess()             // start_process, end_process
    });

  } // for particle

} // function ParticleTable::Fill

} // namespace ng
