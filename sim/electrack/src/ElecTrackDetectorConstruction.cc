// Geant4 exampleB1 copyright statement:
//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: ElecTrackDetectorConstruction.cc $
//
/// \file ElecTrackDetectorConstruction.cc
/// \brief Implementation of the ElecTrackDetectorConstruction class

#include "ElecTrackDetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4Orb.hh"
#include "G4PSDoseDeposit.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4PVPlacement.hh"
#include "G4RunManager.hh"
#include "G4SDManager.hh"
#include "G4ScoringManager.hh"
#include "G4Sphere.hh"
#include "G4SystemOfUnits.hh"
#include "G4Trd.hh"
#include "G4UserLimits.hh"
#include "G4VPrimitiveScorer.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackDetectorConstruction::ElecTrackDetectorConstruction()
    : G4VUserDetectorConstruction(), fStepLimit(NULL), fScoringVolume(0) {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackDetectorConstruction::~ElecTrackDetectorConstruction() {
  delete fStepLimit;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume *ElecTrackDetectorConstruction::Construct() {
  // Get nist material manager
  G4NistManager *nist = G4NistManager::Instance();

  G4SDManager *manager = G4SDManager::GetSDMpointer();

  // Envelope parameters
  //
  G4double env_sizeXY = 20000 * um, env_sizeZ = 20 * um;
  G4Material *env_mat = nist->FindOrBuildMaterial("G4_AIR");

  // Option to switch on/off checking of volumes overlaps
  //
  G4bool checkOverlaps = true;

  //
  // World
  //
  G4double world_sizeXY = 2.2 * env_sizeXY;
  G4double world_sizeZ = 2.2 * env_sizeZ;
  G4Material *world_mat = nist->FindOrBuildMaterial("G4_AIR");

  G4Box *solidWorld = new G4Box("World", // its name
                                0.5 * world_sizeXY, 0.5 * world_sizeXY,
                                0.5 * world_sizeZ); // its size

  G4LogicalVolume *logicWorld = new G4LogicalVolume(solidWorld, // its solid
                                                    world_mat,  // its material
                                                    "World");   // its name

  G4VPhysicalVolume *physWorld =
      new G4PVPlacement(0,               // no rotation
                        G4ThreeVector(), // at (0,0,0)
                        logicWorld,      // its logical volume
                        "World",         // its name
                        0,               // its mother  volume
                        false,           // no boolean operation
                        0,               // copy number
                        checkOverlaps);  // overlaps checking

  //
  // Envelope
  //
  G4Box *solidEnv = new G4Box("Envelope", // its name
                              0.5 * env_sizeXY, 0.5 * env_sizeXY,
                              0.5 * env_sizeZ); // its size

  G4LogicalVolume *logicEnv = new G4LogicalVolume(solidEnv,    // its solid
                                                  env_mat,     // its material
                                                  "Envelope"); // its name

  new G4PVPlacement(0, // no rotation
                    G4ThreeVector(0.5 * env_sizeXY, 0.5 * env_sizeXY,
                                  0.0), // corner at (0,0,0) in world volume
                    logicEnv,           // its logical volume
                    "Envelope",         // its name
                    logicWorld,         // its mother  volume
                    false,              // no boolean operation
                    0,                  // copy number
                    checkOverlaps);     // overlaps checking

  G4double maxStep = 0.1 * um;
  fStepLimit = new G4UserLimits(maxStep);

  logicWorld->SetUserLimits(fStepLimit);
  logicEnv->SetUserLimits(fStepLimit);
  //
  // Layer 1
  //

  G4double fracMass;
  G4int ncomp;

  G4Element *elSi = new G4Element("Silicon", "Si", 14., 28.066 * g / mole);
  G4Element *elO = new G4Element("Oxygen", "O", 8., 16.0 * g / mole);
  G4Element *elAl = new G4Element("Aluminium", "Al", 13., 27.981 * g / mole);
  G4Element *elN = new G4Element("Nitrogen", "N", 7., 14. * g / mole);
  G4Material *SiO2 = new G4Material("SiO2", 2.2 * g / cm3, ncomp = 2);
  SiO2->AddElement(elSi, fracMass = 46.7435 * perCent);
  SiO2->AddElement(elO, fracMass = 53.2565 * perCent);
  G4Material *Al = new G4Material("Al", 2.7 * g / cm3, ncomp = 1);
  Al->AddElement(elAl, fracMass = 100 * perCent);
  G4Material *Si3N4 = new G4Material("Si3N4", 3.2 * g / cm3, ncomp = 2);
  Si3N4->AddElement(elSi, fracMass = 60.0617 * perCent);
  Si3N4->AddElement(elN, fracMass = 39.9383 * perCent);
  G4Material *Si = new G4Material("Si", 2.33 * g / cm3, ncomp = 1);
  Si->AddElement(elSi, fracMass = 100 * perCent);

  G4Material *mats[13] = {SiO2, Si3N4, SiO2, Al,   SiO2, Al, SiO2,
                          Al,   SiO2,  Al,   SiO2, Si,   Si};

  // K2, 20 um thick total
  // G4double widths[13] = {.3*um, .125*um, .1*um, .58*um, .4*um, .395*um,
  // .3*um, .395*um, .3*um, .395*um, .58*um, 5*um, 11.13*um};

  // Thinned type 4um epi with pizza
  G4double widths[13] = {.3 * um,   .125 * um, .1 * um,   .58 * um, .4 * um,
                         .395 * um, .3 * um,   .395 * um, .3 * um,  .395 * um,
                         .58 * um,  4 * um,    0.200 * um};
  std::string labels[13] = {"Layer0",  "Layer1",  "Layer2", "Layer3", "Layer4",
                            "Layer5",  "Layer6",  "Layer7", "Layer8", "Layer9",
                            "Layer10", "Layer11", "Layer12"};
  std::string sdlabels[13] = {"sd0", "sd1", "sd2", "sd3",  "sd4",  "sd5", "sd6",
                              "sd7", "sd8", "sd9", "sd10", "sd11", "sd12"};
  std::string edlabels[13] = {"ed0", "ed1", "ed2", "ed3",  "ed4",  "ed5", "ed6",
                              "ed7", "ed8", "ed9", "ed10", "ed11", "ed12"};
  G4Box *layers[13];
  G4LogicalVolume *logic_layers[13];
  int index = 13;
  G4double p_height = env_sizeXY;
  G4double p_length = env_sizeXY;
  G4double p_width = env_sizeZ;

  G4double area_used = 0 * um;
  for (int i = 0; i < index; i++) {
    G4Material *layer_mat = mats[i];
    G4ThreeVector pos =
        G4ThreeVector(0, 0, -0.5 * (p_width - widths[i]) + area_used);

    G4Box *layer =
        new G4Box(labels[i],                                        // its name
                  0.5 * p_length, 0.5 * p_height, 0.5 * widths[i]); // its size

    layers[i] = layer;

    G4LogicalVolume *logic_layer =
        new G4LogicalVolume(layers[i],  // its solid
                            layer_mat,  // its material
                            labels[i]); // its name

    logic_layer->SetUserLimits(fStepLimit);
    logic_layers[i] = logic_layer;

    new G4PVPlacement(0, // no rotation
                      pos,
                      logic_layers[i], // its logical volume
                      labels[i],       // its name
                      logicEnv,        // its mother  volume
                      true,            // no boolean operation
                      0,               // copy number
                      checkOverlaps);  // overlaps checking
    area_used = area_used + widths[i];
  }

  //
  // always return the physical World
  //
  fScoringVolume = logic_layers[11];

  return physWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ElecTrackDetectorConstruction::SetMaxStep(G4double maxStep) {
  if ((fStepLimit) && (maxStep > 0.)) {
    fStepLimit->SetMaxAllowedStep(maxStep);
  }
}
