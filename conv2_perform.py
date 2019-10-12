import time
import conv2
import numpy as np
import matplotlib.pyplot as plt

time_taken = []
mat_dims = []
mat_dim = 50
ker_dim = 7
kernel = np.ones((ker_dim, ker_dim)) / (ker_dim * ker_dim)

for i in range(20):
    mat_dim = i * 50 + mat_dim
    mat = np.array(range(mat_dim * mat_dim)).astype(float).reshape((mat_dim, mat_dim))
    mat_dims.append(mat_dim)
    tt = []
    
    t0 = time.process_time()
    A = conv2.conv2_naive(mat, kernel).round().astype(int)
    tt.append(time.process_time() - t0)
    
    t0 = time.process_time()
    B = conv2.conv2_normal(mat, kernel).round().astype(int)
    tt.append(time.process_time() - t0)
    
    t0 = time.process_time()
    C = conv2.conv2_optimized(mat, kernel).round().astype(int)
    tt.append(time.process_time() - t0)
    
    time_taken.append(tt)

for i in range(3):
    plt.plot(mat_dims, [x[i] for x in time_taken])
plt.xlabel('matrix dimension')
plt.ylabel('time taken(/s)')
plt.legend(['naive', 'normal', 'optimized'])
plt.show()