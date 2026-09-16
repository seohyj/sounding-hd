import os
import numpy as np
import matrix_data_pb2

_HERE = os.path.dirname(os.path.abspath(__file__))
PB_FILE_PATH = os.path.join(_HERE, 'inception3_projection_matrix_data.pb')
OUTPUT_NPY_PATH = os.path.join(_HERE, 'pca_matrix.npy')

def convert_pb_to_npy(pb_path, npy_path):
    """
    Reads a MatrixData .pb file, extracts the float matrix, and saves it
    as a .npy file. This version works with the simplified .proto definition.
    """
    try:
        matrix_data = matrix_data_pb2.MatrixData()

        with open(pb_path, 'rb') as f:
            matrix_data.ParseFromString(f.read())

        if len(matrix_data.float_feature) > 0:
            matrix_values = matrix_data.float_feature
            
            known_shape = (2048, 1024)
            
            matrix = np.array(matrix_values).reshape(known_shape)

            if matrix.size != 2048 * 1024:
                print(f"Error: The number of elements ({matrix.size}) does not match the expected size of 2097152.")
                return

            np.save(npy_path, matrix)
            
            print("="*50)
            print("PCA matrix conversion successful!")
            print(f"Original file: {pb_path}")
            print(f"Saved .npy file to: {npy_path}")
            print(f"Matrix shape: {matrix.shape}")
            print("="*50)
        else:
            print("Error: The .pb file does not contain 'float_feature' data, or it's empty.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    if 'PATH HERE' in [PB_FILE_PATH, OUTPUT_NPY_PATH]:
        print("Error: Please fill in the PB_FILE_PATH and OUTPUT_NPY_PATH variables.")
    else:
        convert_pb_to_npy(PB_FILE_PATH, OUTPUT_NPY_PATH)
