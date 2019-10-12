from functools import wraps
import numpy as np

def func_wrapper(func):
    @wraps(func)
    def wrapper(mat, kernel):
        '''
        ==Parameters:
        mat : numpy array, the image to be convoluted
        kernel : numpy array, the convolution kernel/mask/filter

        ==Return:
        output_mat : numpy array, convoluted image
        '''
        if type(mat) != np.ndarray or type(kernel) != np.ndarray:
            raise TypeError('the matrix and kernel must be a numpy array.')
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError('the height and width of the kernel must be the same.')
        if kernel.shape[0] % 2 == 0:
            raise ValueError('the kernel dimension must be an odd number.')
        if kernel.shape[0] > mat.shape[0] or kernel.shape[1] > mat.shape[0]:
            raise ValueError('the matrix must be larger than the kernel.')
        return func(mat, kernel)
    return wrapper

#4 loops - the naive way
@func_wrapper
def conv2_naive(mat, kernel):
    m_rows, m_cols = mat.shape
    k_size = kernel.shape[0]
    s_orig = k_size // 2
    output_mat = np.zeros(mat.shape)
    for i in range(m_rows):
        for j in range(m_cols):
            for k_row in range(k_size):
                s_i = i - s_orig + k_row
                if s_i < 0:
                    continue
                elif s_i >= m_rows:
                    break
                for k_col in range(k_size):
                    s_j = j - s_orig + k_col
                    if s_j < 0:
                        continue
                    elif s_j >= m_cols:
                        break
                    output_mat[i, j] += kernel[k_row, k_col] * mat[s_i, s_j]
    return output_mat

#2 loops - the usual way
@func_wrapper
def conv2_normal(mat, kernel):
    m_rows, m_cols = mat.shape
    k_size = kernel.shape[0]
    s_orig = k_size // 2
    output_mat = np.zeros(mat.shape)
    for i in range(m_rows):
        for j in range(m_cols):
            m_s_i, m_e_i = max(0, i - s_orig), min(m_rows, i + s_orig + 1)
            m_s_j, m_e_j = max(0, j - s_orig), min(m_cols, j + s_orig + 1)
            k_s_i, k_e_i = max(0, s_orig - i), min(k_size, s_orig + m_rows - i)
            k_s_j, k_e_j = max(0, s_orig - j), min(k_size, s_orig + m_cols - j)
            output_mat[i, j] = np.sum(np.sum(mat[m_s_i : m_e_i, m_s_j : m_e_j] * \
                                             kernel[k_s_i : k_e_i, k_s_j : k_e_j]))
    return output_mat

#optimized 1 loop - iterating through the filter instead of the image itself
@func_wrapper
def conv2_optimized(mat, kernel):
    m_rows, m_cols = mat.shape
    k_size = kernel.shape[0]
    s_orig = k_size // 2
    output_mat = np.zeros(mat.shape)
    for k in range(k_size * k_size):
        k_i, k_j = divmod(k, k_size)
        
        m_s_i, m_e_i = max(0, s_orig - k_i), min(m_rows, m_rows + s_orig - k_i)
        m_s_j, m_e_j = max(0, s_orig - k_j), min(m_cols, m_cols + s_orig - k_j)
        k_s_i, k_e_i = max(0, k_i - s_orig), min(m_rows, m_rows + k_i - s_orig)
        k_s_j, k_e_j = max(0, k_j - s_orig), min(m_cols, m_cols + k_j - s_orig)
        output_mat[m_s_i : m_e_i, m_s_j : m_e_j] =  output_mat[m_s_i : m_e_i, m_s_j : m_e_j] + \
                                                    kernel[k_i, k_j] * mat[k_s_i : k_e_i, k_s_j : k_e_j]
    return output_mat

#testing
if __name__ == '__main__':
    mat = np.array(range(25)).astype(float).reshape((5, 5))
    kernel = np.ones((3, 3)) / 9
    A = conv2_naive(mat, kernel)
    B = conv2_normal(mat, kernel)
    C = conv2_optimized(mat, kernel)
    print(A)
    print(B)
    print(C)