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
// $Id: ElecTrackActionInitialization.cc $
//
/// \file ElecTrackActionInitialization.cc
/// \brief Implementation of the ElecTrackActionInitialization class

#include "ElecTrackActionInitialization.hh"
#include "ElecTrackEventAction.hh"
#include "ElecTrackPrimaryGeneratorAction.hh"
#include "ElecTrackRunAction.hh"
#include "ElecTrackSteppingAction.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackActionInitialization::ElecTrackActionInitialization()
    : G4VUserActionInitialization() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ElecTrackActionInitialization::~ElecTrackActionInitialization() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ElecTrackActionInitialization::BuildForMaster() const {
  ElecTrackRunAction *runAction = new ElecTrackRunAction;
  SetUserAction(runAction);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ElecTrackActionInitialization::Build() const {
  SetUserAction(new ElecTrackPrimaryGeneratorAction);

  ElecTrackRunAction *runAction = new ElecTrackRunAction;
  SetUserAction(runAction);

  ElecTrackEventAction *eventAction = new ElecTrackEventAction(runAction);
  SetUserAction(eventAction);

  SetUserAction(new ElecTrackSteppingAction(eventAction));
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
