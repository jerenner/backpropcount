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
// $Id: ElecTrackPrimaryGeneratorAction.cc $
//
/// \file ElecTrackPrimaryGeneratorAction.cc
/// \brief Implementation of the ElecTrackPrimaryGeneratorAction class

#include "ElecTrackPrimaryGeneratorAction.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <fstream>
extern std::ofstream outfile;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackPrimaryGeneratorAction::ElecTrackPrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(), fParticleGun(0), fEnvelopeBox(0) {
  G4int n_particle = 1;
  fParticleGun = new G4ParticleGun(n_particle);

  // default particle kinematic
  G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  G4ParticleDefinition *particle =
      particleTable->FindParticle(particleName = "e-");
  fParticleGun->SetParticleDefinition(particle);
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));

  // Set particle energy
  fParticleGun->SetParticleEnergy(300. * keV);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackPrimaryGeneratorAction::~ElecTrackPrimaryGeneratorAction() {
  delete fParticleGun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ElecTrackPrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent) {
  // this function is called at the begining of ecah event
  //

  // In order to avoid dependence of PrimaryGeneratorAction
  // on DetectorConstruction class we get Envelope volume
  // from G4LogicalVolumeStore.

  G4double envSizeXY = 0;
  G4double envSizeZ = 0;

  if (!fEnvelopeBox) {
    G4LogicalVolume *envLV =
        G4LogicalVolumeStore::GetInstance()->GetVolume("Envelope");
    if (envLV)
      fEnvelopeBox = dynamic_cast<G4Box *>(envLV->GetSolid());
  }

  if (fEnvelopeBox) {
    envSizeXY = fEnvelopeBox->GetXHalfLength() * 2.;
    envSizeZ = fEnvelopeBox->GetZHalfLength() * 2.;
  } else {
    G4ExceptionDescription msg;
    msg << "Envelope volume of box shape not found.\n";
    msg << "Perhaps you have changed geometry.\n";
    msg << "The gun will be place at the center.";
    G4Exception("ElecTrackPrimaryGeneratorAction::GeneratePrimaries()",
                "MyCode0002", JustWarning, msg);
  }

  G4double size = 0.98;
  G4double pixel_size = 5.0 * um;
  G4bool front_side = false;

  G4double x0 = (envSizeXY / 2) *
                (1 + size * (2 * G4UniformRand() -
                             1)); // default for uniform over the geometry area
  G4double y0 = (envSizeXY / 2) *
                (1 + size * (2 * G4UniformRand() -
                             1)); // default for uniform over the geometry area

  // Generate in the center of the sensor over 1 pixel width
  // G4double x0 = (envSizeXY / 2) + (pixel_size / 2) * (2 * G4UniformRand() -
  // 1); G4double y0 = (envSizeXY / 2) + (pixel_size / 2) * (2 * G4UniformRand()
  // - 1);

  G4double z0;
  if (front_side) {
    z0 = -0.5 * envSizeZ; // default (front side)
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
  } else {
    z0 = 0.5 * envSizeZ; // from the other (back) side
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., -1.));
  }

  fParticleGun->SetParticlePosition(
      G4ThreeVector(x0, y0, z0)); // default random over the area

  outfile << "new_e- " << x0 << " " << y0 << " " << z0 << " "
          << fParticleGun->GetParticleMomentumDirection()[2] << " "
          << fParticleGun->GetParticleEnergy() << "\n";

  fParticleGun->GeneratePrimaryVertex(anEvent);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
