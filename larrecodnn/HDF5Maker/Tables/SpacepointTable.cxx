#include "larrecodnn/HDF5Maker/Tables/SpacepointTable.h"

#include "canvas/Persistency/Common/FindManyP.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/SpacePoint.h"

namespace ng {

  //-----------------------------------------------------------------------------
  // names of columns in spacepoint table
  std::vector<std::string> static const SpacepointColumns{"run",
                                                          "subrun",
                                                          "event",
                                                          "sp_id",
                                                          "position_x",
                                                          "position_y",
                                                          "position_z",
                                                          "wh_id_1",
                                                          "wh_id_2",
                                                          "wh_id_3"};

  //-----------------------------------------------------------------------------
  // spacepoint table constructor
  SpacepointTable::SpacepointTable(std::string const& spacepointLabel, std::vector<Row> const& data)
    : Table("spacepoints", SpacepointColumns, data), fSpacepointLabel(spacepointLabel)
  {}

  //-----------------------------------------------------------------------------
  void SpacepointTable::Fill(art::Event const& evt)
  {
    // get event ID
    art::EventID const& id = evt.id();

    auto sps = evt.getHandle<std::vector<recob::SpacePoint>>(fSpacepointLabel);
    art::FindManyP<recob::Hit> fmp(sps, evt, fSpacepointLabel);

    // loop over spacepoints
    for (unsigned int sp_id = 0; sp_id < sps->size(); ++sp_id) {
      recob::SpacePoint const& sp = sps->at(sp_id);

      // store IDs of associated hits
      std::array<int, 3> wh_id = {-1, -1, -1};
      for (art::Ptr<recob::Hit> const& hit : fmp.at(sp_id)) {
        wh_id[hit->View()] = hit.key();
      } // for hit

      fData.push_back({
        id.run(),
        id.subRun(),
        id.event(),
        sp_id, // event ID, sp_id
        sp.XYZ()[0],
        sp.XYZ()[1],
        sp.XYZ()[2], // position
        wh_id[0],
        wh_id[1],
        wh_id[2] // wire hit IDs
      });

    } // for spacepoint

  } // function SpacepointTable::Fill

} // namespace ng
