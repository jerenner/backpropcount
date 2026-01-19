#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

const int PIXEL_ARRAY_SIZE = 4000;
const double PIXEL_SIZE_UM = 5.0;

// function to split string into vector of strings
vector<string> split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str); // Turn the string into a stream.
  string tok;

  while (getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }

  return internal;
}

int main(int argc, char **argv) {

  double posx, posy, posz;
  double xi, yi, zi; // the current position of the ionization electron
  double de, total_e;
  int n_electrons;
  double costheta, phi, step_length;
  int **pixel_array = new int *[PIXEL_ARRAY_SIZE];
  for (int i = 0; i < PIXEL_ARRAY_SIZE; i++)
    pixel_array[i] = new int[PIXEL_ARRAY_SIZE];

  double pixel_size = PIXEL_SIZE_UM; // um in x and y
  double pair_energy = 3.6;          // avergage energy to make e-h pair in eV
  double top_of_epi = -6.2;          // in um
  double bottom_of_epi = -2.2;       // in um for the 4um EPI thinned
  // double bottom_of_epi = -1.2; // in um for the 5um K2 EPI

  double thermal_electron_step =
      1.0; // in um the average step of a thermal electron before collision and
           // effective randomization

  std::default_random_engine generator;

  std::exponential_distribution<double> distribution_step(
      thermal_electron_step);
  std::uniform_real_distribution<double> distribution_costheta(-1., 1.);
  std::uniform_real_distribution<double> distribution_phi(0., 2.0 * 3.1415926);

  // read all the lines from the input file
  int read_events = 0;
  string line;

  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <input_file>" << endl;
    return 1;
  }
  string input_filename = argv[1];
  ifstream input_file(input_filename);

  string output_filename =
      "pixelated_" +
      input_filename.substr(input_filename.find_last_of("/\\") + 1);
  ofstream output_file;
  output_file.open(output_filename);
  output_file << "# " << PIXEL_ARRAY_SIZE << " " << PIXEL_ARRAY_SIZE << " "
              << PIXEL_ARRAY_SIZE * PIXEL_SIZE_UM << " "
              << PIXEL_ARRAY_SIZE * PIXEL_SIZE_UM << endl;

  if (!input_file.is_open()) {
    cout << "Error opening file: " << input_filename << endl;
    return 1;
  }

  while (getline(input_file, line)) {

    // check if header line or track point

    vector<string> sep = split(line, ' ');
    if (sep[0] == "new_e-") {

      read_events++;
      if (read_events > 1) {
        // write the pixel matrix for previous event
        // completed tracing all electrons now write out and rezero array
        // in sparse mode
        for (int indy = 0; indy < PIXEL_ARRAY_SIZE; indy++) {
          for (int indx = 0; indx < PIXEL_ARRAY_SIZE; indx++) {
            if (pixel_array[indy][indx] != 0) {
              output_file << indx << " " << indy << " "
                          << pixel_array[indy][indx] << '\n';
              // zero array element
              pixel_array[indy][indx] = 0;
            }
          }
        }
        // and rezero the matrix
        total_e = 0.;
      }

      if (read_events % 100 == 0) {
        cout << "Event number: " << read_events << '\n';
      }

      // event header has event number, x and y of true position direction and
      // primary energy
      output_file << "EV" << " " << (read_events - 1) << " " << sep[1] << " "
                  << sep[2] << " " << sep[4] << " " << sep[5] << '\n';

    } else if (sep[0] == "TEST") {
      // ignore these lines
    } else {
      // is a point in a track
      // posxyz are the coordinates of the energy deposition in mm, here
      // converted to um
      posx = stod(sep[0]) * 1000.;
      posy = stod(sep[1]) * 1000.;
      posz = stod(sep[2]) * 1000.;
      // the energy lost inthe step in MeV units
      de = stod(sep[3]);
      total_e += de;
      // for now track individual electrons without matrices/binning
      n_electrons = de * 1e6 / pair_energy;

      // loop over the ionization electrons produced in this step
      while (n_electrons > 0) {

        xi = posx;
        yi = posy;
        zi = posz; // initialize

        while (zi > top_of_epi) { // keep tracking this electron as long as it
                                  // hasn't reached the top of epi
          // a fancier simulation would include x-y geometry of the diode and
          // would make other areas reflective get a random direction
          step_length = distribution_step(generator);
          costheta = distribution_costheta(generator);
          phi = distribution_phi(generator);

          xi += step_length * cos(phi) * sqrt(1.0 - costheta * costheta);
          yi += step_length * sin(phi) * sqrt(1.0 - costheta * costheta);
          zi += step_length * costheta;

          if (zi > bottom_of_epi) { // bouncing like a mirror from the EPI
                                    // substrate interface
            zi = bottom_of_epi - (zi - bottom_of_epi);
          }
        }
        // ionization electron has reached the top of EPI
        // find the position where it
        // cout << "hit the top of the EPI " << " " << n_electrons << '\n';
        // put in charge matrix, eventually refine to find the crossing point
        // of EPI top for now report first position above EPI
        int ii = int(yi / pixel_size);
        int jj = int(xi / pixel_size);

        if (ii > PIXEL_ARRAY_SIZE - 1 || jj > PIXEL_ARRAY_SIZE - 1 || ii < 0 ||
            jj < 0) {
          cout << "WARNING, pixel outside of range not added to array: ii: "
               << ii << " jj: " << jj << " for yi = " << yi
               << " and xi = " << xi << endl;
        } else {
          pixel_array[ii][jj]++;
        }
        n_electrons--;
      }
    }
  }

  // write the pixel matrix for the final event
  if (read_events > 1) {
    for (int indy = 0; indy < PIXEL_ARRAY_SIZE; indy++) {
      for (int indx = 0; indx < PIXEL_ARRAY_SIZE; indx++) {
        if (pixel_array[indy][indx] != 0) {
          output_file << indx << " " << indy << " " << pixel_array[indy][indx]
                      << '\n';
          // zero array element
          pixel_array[indy][indx] = 0;
        }
      }
    }
  }
  input_file.close();
  output_file.close();

  // Delete the dynamically allocated pixel array
  for (int i = 0; i < PIXEL_ARRAY_SIZE; ++i)
    delete[] pixel_array[i];
  delete[] pixel_array;
}
