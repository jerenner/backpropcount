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
// $Id: ElecTrackSteppingAction.cc $
//
/// \file ElecTrackSteppingAction.cc
/// \brief Implementation of the ElecTrackSteppingAction class

#include "ElecTrackSteppingAction.hh"
#include "ElecTrackDetectorConstruction.hh"
#include "ElecTrackEventAction.hh"

#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include <fstream>
extern std::ofstream outfile;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackSteppingAction::ElecTrackSteppingAction(
    ElecTrackEventAction *eventAction)
    : G4UserSteppingAction(), fEventAction(eventAction), fScoringVolume(0) {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackSteppingAction::~ElecTrackSteppingAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ElecTrackSteppingAction::UserSteppingAction(const G4Step *step) {
  if (!fScoringVolume) {
    const ElecTrackDetectorConstruction *detectorConstruction =
        static_cast<const ElecTrackDetectorConstruction *>(
            G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolume = detectorConstruction->GetScoringVolume();
  }

  // get volume of the current step
  G4LogicalVolume *volume = step->GetPreStepPoint()
                                ->GetTouchableHandle()
                                ->GetVolume()
                                ->GetLogicalVolume();
  G4StepPoint *preStep = step->GetPreStepPoint();
  const G4Event *evt = G4RunManager::GetRunManager()->GetCurrentEvent();
  G4int eid2 = evt->GetEventID();
  if (eid2 != eid) {
    x0 = 100;
    y0 = 100;
    eid = evt->GetEventID();
  }
  if (x0 == 100)
    x0 = preStep->GetPosition().x();
  if (y0 == 100)
    y0 = preStep->GetPosition().y();
  // check if we are in scoring volume
  if (volume != fScoringVolume)
    return;

  // collect energy deposited in this step
  G4double edepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddEdep(edepStep);

  G4double posx = preStep->GetPosition().x();
  G4double posy = preStep->GetPosition().y();
  G4double posz = preStep->GetPosition().z();
  eid = evt->GetEventID();

  // Output all energy depositions.
  // outfile is already opened at this point
  outfile << posx << " " << posy << " " << posz << " " << edepStep << " " << eid
          << " " << x0 << " " << y0 << "\n";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
