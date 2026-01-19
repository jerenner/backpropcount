import pandas as pd
import sys
import os

def read_electron_data(fname, pixel_size=10.0, nevts=1000000000):

    evt = -1
    xinc = 0.0
    yinc = 0.0
    front = True
    energy = 0.0

    # Open the file and read the specified number of events.
    l_evt, l_xinc, l_yinc, l_front, l_energy, l_row, l_col, l_counts = [], [], [], [], [], [], [], []
    evts_read = 0
    with open(fname) as f:

        # Iterate through all lines.
        for line in f:
            
            # Skip comments
            if line.startswith("#"):
                continue

            # Stop reading if we've read the specified number of events.
            if(evts_read > nevts):
                break

            # Get each number in the line, separated by spaces.
            vals = line.rstrip().split(" ")
            
            if len(vals) == 0:
                continue

            # Start a new event.
            if(vals[0] == "EV"):
                evt    = vals[1]
                xinc   = vals[2]
                yinc   = vals[3]
                front  = (vals[4] == 1)
                energy = vals[5]
                evts_read += 1

            # Add a row for the current event.

            else:
                try:
                    # Calculate central pixel coordinates (rounded to nearest integer)
                    # Assuming xinc/yinc are in mm and pixel_size is in um
                    center_row = int(float(xinc) * 1000 / pixel_size)
                    center_col = int(float(yinc) * 1000 / pixel_size)

                    l_evt.append(int(evt))
                    l_xinc.append(float(xinc))
                    l_yinc.append(float(yinc))
                    l_front.append(front)
                    l_energy.append(float(energy))
                    
                    # Center the event
                    row_idx = int(vals[0]) - center_row + 50
                    col_idx = int(vals[1]) - center_col + 50
                    
                    l_row.append(row_idx)
                    l_col.append(col_idx)
                    l_counts.append(int(vals[2]))
                except ValueError:
                    continue

    # Construct the DataFrame.
    evt_dict = {'event': l_evt, 'xinc': l_xinc, 'yinc': l_yinc, 'front': l_front,  # x and y incidence points (true x and y points that led to the pixels; subpixel numbers)
                'energy': l_energy, 'row': l_row, 'col': l_col, 'counts': l_counts}
    df = pd.DataFrame.from_dict(evt_dict)

    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pixelated_file> <pixel_size>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    print(f"Processing {input_file}...")
    
    pixel_size = 10.0
    try:
        pixel_size = float(sys.argv[2])
        print(f"Using pixel size: {pixel_size} um")
    except ValueError:
        print(f"Error: Invalid pixel size '{sys.argv[2]}'. Using default 10.0 um.")

    df = read_electron_data(input_file, pixel_size=pixel_size)
    
    output_file = os.path.splitext(input_file)[0] + ".pkl"
    print(f"Saving to {output_file}...")
    df.to_pickle(output_file)
    print("Done.")

