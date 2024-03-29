////////////////////////////////////////////////////////////////////////
// \file    Interaction.h
///\brief   Defines an enumeration for interaction type
///
// \author   radovic -- a.radovic@gmail.com
//           Saul Alonso Monsalve -- saul.alonso.monsalve@cern.ch
////////////////////////////////////////////////////////////////////////
#ifndef LCVN_INTERACTION_H
#define LCVN_INTERACTION_H

#include "larrecodnn/CVN/func/PixelMap.h"

namespace lcvn {

  enum InteractionType {
    kNumuQE,            ///< Numu CC QE interaction
    kNumuRes,           ///< Numu CC Resonant interaction
    kNumuDIS,           ///< Numu CC DIS interaction
    kNumuOther,         ///< Numu CC, other than above
    kNueQE,             ///< Nue CC QE interaction
    kNueRes,            ///< Nue CC Resonant interaction
    kNueDIS,            ///< Nue CC DIS interaction
    kNueOther,          ///< Nue CC, other than above
    kNutauQE,           ///< Nutau CC QE interaction
    kNutauRes,          ///< Nutau CC Resonant interaction
    kNutauDIS,          ///< Nutau CC DIS interaction
    kNutauOther,        ///< Nutau CC, other than above
    kNuElectronElastic, ///< NC Nu On E Scattering
    kNC,                ///< NC interaction
    kCosmic,            ///< Cosmic ray background
    kOther,             ///< Something else.  Tau?  Hopefully we don't use this
    kNIntType           ///< Number of interaction types, used like a vector size
  };

  /// Enumeration to describe the order of the TF network output
  enum TFResultType {
    kTFNumuQE,     ///< Numu CC QE interaction
    kTFNumuRes,    ///< Numu CC Resonant interaction
    kTFNumuDIS,    ///< Numu CC DIS interaction
    kTFNumuOther,  ///< Numu CC, other than above
    kTFNueQE,      ///< Nue CC QE interaction
    kTFNueRes,     ///< Nue CC Resonant interaction
    kTFNueDIS,     ///< Nue CC DIS interaction
    kTFNueOther,   ///< Nue CC, other than above
    kTFNutauQE,    ///< Nutau CC QE interaction
    kTFNutauRes,   ///< Nutau CC Resonant interaction
    kTFNutauDIS,   ///< Nutau CC DIS interaction
    kTFNutauOther, ///< Nutau CC, other than above
    kTFNC          ///< NC interaction
  };

  // Enumeration to describe the different outputs of the TF multioutput network
  enum TFMultioutputs { is_antineutrino, flavour, interaction, protons, pions, pizeros, neutrons };

  // Enumeration to describe the is_antineutrino-reduced TF multioutput network
  enum TFIsAntineutrino { kNeutrino, kAntineutrino };

  // Enumeration to describe the flavour-reduced TF multioutput network
  enum TFFlavour { kFlavNumuCC, kFlavNueCC, kFlavNutauCC, kFlavNC };

  // Enumeration to describe the interaction-reduced TF multioutput network
  enum TFInteraction { kInteQECC, kInteResCC, kInteDISCC, kInteOtherCC };

  // Enumeration to describe the protons (topology information) TF multioutput network
  enum TFTopologyProtons { kTop0proton, kTop1proton, kTop2proton, kTopNproton };

  // Enumeration to describe the pions (topology information) TF multioutput network
  enum TFTopologyPions { kTop0pion, kTop1pion, kTop2pion, kTopNpion };

  // Enumeration to describe the pizeros (topology information) TF multioutput network
  enum TFTopologyPizeros { kTop0pizero, kTop1pizero, kTop2pizero, kTopNpizero };

  // Enumeration to describe the neutrons (topology information) TF multioutput network
  enum TFTopologyNeutrons { kTop0neutron, kTop1neutron, kTop2neutron, kTopNneutron };

  // // It might be a good idea to consider topology information.
  // // We should use base two here so we can & and | results.
  // enum Topology
  // {
  //   kTopUnset    = 0x000000,
  //   // Flavours
  //   kTopNumu     = 0x000001,
  //   kTopNue      = 0x000002,
  //   kTopNutau    = 0x000004,
  //   kTopNC       = 0x000008,
  //   // Topologies - protons
  //   kTop0proton  = 0x000010,
  //   kTop1proton  = 0x000020,
  //   kTop2proton  = 0x000040,
  //   kTopNproton  = 0x000080,
  //   // Topologies - pions
  //   kTop0pion    = 0x000100,
  //   kTop1pion    = 0x000200,
  //   kTop2pion    = 0x000400,
  //   kTopNpion    = 0x000800,
  //   // Topologies - pizeros
  //   kTop0pizero  = 0x001000,
  //   kTop1pizero  = 0x002000,
  //   kTop2pizero  = 0x004000,
  //   kTopNpizero  = 0x008000,
  //   // Topologies - neutrons
  //   kTop0neutron = 0x010000,
  //   kTop1neutron = 0x020000,
  //   kTop2neutron = 0x040000,
  //   kTopNneutron = 0x080000,
  //   // Topologies - tau
  //   kTopNotTau   = 0x100000,
  //   kTopTauHad   = 0x200000,
  //   kTopTauE     = 0x400000,
  //   kTopTauMu    = 0x800000,

  //   // Antineutrino
  //   kTopIsAntiNeutrino = 0x1000000

  // };

  enum TauType { kNotNutau, kNutauE, kNutauMu, kNutauHad };

  enum TopologyType { kTopNue, kTopNumu, kTopNutauE, kTopNutauMu, kTopNutauHad, kTopNC };

  enum TopologyTypeAlt { kTopNueLike, kTopNumuLike, kTopNutauLike, kTopNCLike };

}

#endif // CVN_INTERACTIONTYPE_H
