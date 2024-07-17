import os
import json
import numpy as np

def json_to_npy(json_file, npy_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    npy_data = []

    for frame in frames:
        # Append the transformation matrix
        flat_transform = np.array(frame['transform_matrix'])

        # Flatten the 4x4 transformation matrix to a 1D list and select the first 12 elements
        flat_transform = flat_transform.flatten()[:12].tolist()

        # Append the extra values [0, 0, 0]
        flat_transform.extend([0.0, 0.0, 0.0])

        # Append the bounds [2, 6]
        flat_transform.extend([2.0, 6.0])

        npy_data.append(flat_transform)

    # Convert the list to a numpy array
    npy_array = np.array(npy_data)

    # Save to .npy file
    np.save(npy_file, npy_array)

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.curdir, "data", "ship")
    json_file = os.path.join(dataset_path, "transforms_train.json")
    npy_file = os.path.join(dataset_path, "poses_bounds.npy")
    json_to_npy(json_file, npy_file)
