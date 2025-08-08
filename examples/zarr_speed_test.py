import zarr
import time
import numpy as np
import sys
import os
import shutil


# Create zarr file to store array

def create_zarr_file(file_path, shape, chunks):
    zarr_array = zarr.zeros(shape, chunks=chunks, dtype='f8', store=file_path)#, compressor=None)
    return zarr_array


def read_zarr_file(file_path):
    #ids = np.random.randint(0, 512, size=512)
    zarr_array2 = zarr.open(file_path, mode='r')
    for i in range(128):
        zarr_array = zarr_array2[i:i+1][:]
        print(zarr_array.shape)
    return zarr_array

def write_data_to_zarr(zarr_array, data):
    zarr_array[:] = data

def main(file_path):
    shape = (128, 128, 128, 128)
    chunks = (1,)

    # Create random data to write
    timeA = time.time()
    data = np.random.rand(*shape).astype('f8')
    #data = np.zeros(shape).astype('f8')

    # Create a Zarr file
    time0 = time.time()
    zarr_array = create_zarr_file(file_path, shape, chunks)

    # Write some data to the Zarr file
    time1 = time.time()
    write_data_to_zarr(zarr_array, data)
    
    # Read the data back from the Zarr file
    time2 = time.time()
    read_data = read_zarr_file(file_path)
    print("Data read from Zarr file:", read_data.shape)
    time3 = time.time()

    # Print timing results
    print(f"Time to create random data: {time0 - timeA:.4f} seconds")
    print(f"Time to create Zarr file: {time1 - time0:.4f} seconds")
    print(f"Time to write data to Zarr file: {time2 - time1:.4f} seconds")
    print(f"Time to read data from Zarr file: {time3 - time2:.4f} seconds")

    # Clean up
    zarr_array.store.close()

if __name__ == "__main__":
    # Read file directory from command line argument or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'test.zarr'

    main(file_path)

    # Remove the created Zarr file after testing (directory!)
    if os.path.exists(file_path):
        #os.rmdir(file_path)
        shutil.rmtree(file_path)
        print(f"Removed Zarr file: {file_path}")
    else:
        print(f"Zarr file {file_path} does not exist.")
