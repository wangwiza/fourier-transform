import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import argparse
import time

# The naive DFT algorithm
def naive_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    # both k and n go from 0 to N-1
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# The naive IDFT algorithm
def naive_idft(X):
    N = len(X)
    x = np.zeros(N, dtype=np.complex128)
    # both k and n go from 0 to N-1
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

# The Cooley-Tukey FFT algorithm
def fft(x):
    N = len(x)
    if N <= 1:
        return x
    X_even = fft(x[::2])
    X_odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([X_even + factor[:N // 2] * X_odd,
                           X_even + factor[N // 2:] * X_odd])

# Inverse FFT
def ifft(X):
    N = len(X)
    if N <= 1:
        return X
    x_even = ifft(X[::2])
    x_odd = ifft(X[1::2])
    factor = np.exp(2j * np.pi * np.arange(N) / N)
    return np.concatenate([x_even + factor[:N // 2] * x_odd,
                           x_even + factor[N // 2:] * x_odd]) / 2

def fft2(x_matrix):
    """
    Perform a 2D FFT using the provided 1D FFT function.
    This function assumes that x_matrix is a 2D numpy array.
    """
    # Apply 1D FFT along the rows
    row_fft = np.array([fft(row) for row in x_matrix])
    
    # Apply 1D FFT along the columns
    col_fft = np.array([fft(col) for col in row_fft.T]).T
    
    return col_fft

def ifft2(X_matrix):
    """
    Perform a 2D Inverse FFT using the provided 1D IFFT function.
    This function assumes that X_matrix is a 2D numpy array.
    """
    # Apply 1D IFFT along the rows
    row_ifft = np.array([ifft(row) for row in X_matrix])
    # Apply 1D IFFT along the columns
    col_ifft = np.array([ifft(col) for col in row_ifft.T]).T
    # Normalize by the total number of elements (N x M)
    return col_ifft

# Plot FFT
def plot_fft(image, fft_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("FFT of Image")
    plt.imshow(np.abs(fft_image), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.show()

# Denoise Image
def denoise_image(image, keep_fraction=0.1):
    im_fft = fft2(image)
    r, c = im_fft.shape
    # Create a mask to identify which values are zeroed out
    mask = np.ones_like(im_fft, dtype=bool)
    # Set to zero all rows with indices between r*keep_fraction and r*(1-keep_fraction)
    mask[int(r*keep_fraction):int(r*(1-keep_fraction)), :] = False
    # Similarly with the columns
    mask[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = False
    # Apply the mask to zero out high frequencies
    im_fft[~mask] = 0
    # Perform the inverse FFT
    im_new = ifft2(im_fft).real
    # Count the number of zeroed out values
    num_zeroed_out = np.sum(~mask)
    return im_new, num_zeroed_out

# Compress Image


# Plot Compressed Images


# Plot Runtime
def plot_runtime():
    sizes = [2**i for i in range(5, 11)]
    dft_times = []
    fft_times = []
    for size in sizes:
        x = np.random.random(size)
        start_time = time.time()
        naive_dft(x)
        dft_times.append(time.time() - start_time)
        start_time = time.time()
        fft(x)
        fft_times.append(time.time() - start_time)
    plt.plot(sizes, dft_times, label='DFT')
    plt.plot(sizes, fft_times, label='FFT')
    plt.xlabel('Input size')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()

# Pad image to the next power of 2
def pad_image(image):
    rows, cols = image.shape
    new_rows = 2**int(np.ceil(np.log2(rows)))
    new_cols = 2**int(np.ceil(np.log2(cols)))
    padded_image = np.zeros((new_rows, new_cols), dtype=image.dtype)
    padded_image[:rows, :cols] = image
    return padded_image

# Main Function
def main():
    parser = argparse.ArgumentParser(description='Perform FFT on an image.')
    parser.add_argument('-m', '--mode', type=int, default=1, help='Mode of operation')
    parser.add_argument('-i', '--image', type=str, default='moonlanding.png', help='Image file name')
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {args.image}")
        return

    original_shape = image.shape
    image = pad_image(image)


    if args.mode == 1:
        fft_image = fft2(image)
        fft_image = cv2.resize(np.abs(fft_image), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        image = image[:original_shape[0], :original_shape[1]]
        plot_fft(image, fft_image)
    elif args.mode == 2:
        denoised_image, num_zeroed_out = denoise_image(image)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image[:original_shape[0], :original_shape[1]], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Denoised Image")
        plt.imshow(denoised_image[:original_shape[0], :original_shape[1]], cmap='gray')
        total_elements = image.size
        print(f"Number of zeroed out values: {num_zeroed_out}")
        print(f"Fraction of zeroed out values: {num_zeroed_out / total_elements:.4f}")
        plt.show()
    elif args.mode == 3:
        pass
        
    elif args.mode == 4:
        plot_runtime()

if __name__ == '__main__':
    main()
