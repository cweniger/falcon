import ray
import numpy as np
import time
from collections import deque

#ray.init(object_store_memory=10 * 1024 ** 3, ignore_reinit_error=True)
ray.init(address='auto')
quit()

ARRAY_SHAPE = (128, 128, 128)
SLEEP = 1e-1
MAX_OBJECTS = 10_000

object_refs = deque()
count = 0
total_mean = 0.0

print("Array size [MB]:", ARRAY_SHAPE[0] * ARRAY_SHAPE[1] * ARRAY_SHAPE[2] * 8 / 1024 ** 2)

try:
    while len(object_refs) < MAX_OBJECTS:
        # Store array in Ray object store
        array = np.zeros(ARRAY_SHAPE)
        ref = ray.put(array)
        del array
        object_refs.append(ref)

        if len(object_refs) >= 10_000:  # Adjustable batch size
            print(f"Reading back {len(object_refs)} arrays...")
            while object_refs:
                arr = ray.get(object_refs.popleft())
                total_mean += arr.mean()
                count += 1
                del arr
        time.sleep(SLEEP)

except Exception as e:
    print(f"Exception: {e}")

#finally:
    #print(f"Final total mean of {count} arrays: {total_mean:.4f}")
    #ray.shutdown()