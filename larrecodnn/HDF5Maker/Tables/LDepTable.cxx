#include "larrecodnn/HDF5Maker/Tables/LDepTable.h"

#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "larsim/MCCheater/PhotonBackTrackerService.h"

namespace ng {

  //-----------------------------------------------------------------------------
  // names of columns in light deposit table
  std::vector<std::string> static const LDepColumns{"run",
                                                    "subrun",
                                                    "event",
                                                    "oh_id",
                                                    "g4_id",
                                                    "energy",
                                                    "position_x",
                                                    "position_y",
                                                    "position_z"};

  //-----------------------------------------------------------------------------
  // light deposit table constructor
  LDepTable::LDepTable(std::string const& opHitLabel, std::vector<Row> const& data)
    : Table("ldeps", LDepColumns, data), fOpHitLabel(opHitLabel)
  {}

  //-----------------------------------------------------------------------------
  void LDepTable::Fill(art::Event const& evt)
  {
    // get event ID
    art::EventID const& id = evt.id();

    // get service handles
    art::ServiceHandle<cheat::PhotonBackTrackerService> bt;

    // loop over hits
    auto hits = evt.getHandle<std::vector<recob::OpHit>>(fOpHitLabel);
    for (size_t oh_id = 0; oh_id < hits->size(); ++oh_id) {
      recob::OpHit const& hit = hits->at(oh_id);

      // skip events with no track IDs
      if (!bt->OpHitToTrackIds(hit).size()) continue;

      std::vector<double> const xyz = bt->OpHitToXYZ(hit);

      // loop over averaged sim::IDEs
      for (const sim::TrackSDP& sdp : bt->OpHitToEveTrackSDPs(hit)) {

        // catch negative track IDs
        if (sdp.trackID < 0) {
          throw art::Exception(art::errors::LogicError)
            << "Negative track ID (" << sdp.trackID
            << ") found in simulated "
               "light deposits! This is usually an indication that you're "
               "running over simulation from before the larsoft Geant4 "
               "refactor, which is not supported due to its incomplete MC "
               "truth record.";
        } // if track ID is negative

        fData.push_back({
          id.run(),
          id.subRun(),
          id.event(), // event ID
          oh_id,
          sdp.trackID,
          sdp.energy, // oh_id, g4_id, energy
          xyz[0],
          xyz[1],
          xyz[2] // position
        });

      } // for light deposit
    }   // for hit

  } // function LDepTable::Fill

} // namespace ng
