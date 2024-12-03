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

def dft2(x_matrix):
    """
    Perform a 2D DFT using the provided 1D DFT function.
    This function assumes that x_matrix is a 2D numpy array.
    """
    # Apply 1D DFT along the rows
    row_dft = np.array([naive_dft(row) for row in x_matrix])
    
    # Apply 1D DFT along the columns
    col_dft = np.array([naive_dft(col) for col in row_dft.T]).T
    
    return col_dft

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

# Denoise Image (Keep Low Frequencies)
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
    num_non_zeros = np.count_nonzero(mask)
    print(f"Number of non-zero values: {num_non_zeros}")
    print(f"Fraction of non-zero values: {num_non_zeros / im_fft.size:.4f}")
    # Perform the inverse FFT
    im_new = ifft2(im_fft).real
    # Count the number of zeroed out values
    return im_new

# Denoise Image (Thresholding)
def denoise_image_threshold(image, threshold=0.5):
    # Perform the FFT
    im_fft = fft2(image)
    # Get the magnitude of the FFT
    im_fft_magnitude = np.abs(im_fft)
    # Determine the threshold value
    max_magnitude = np.max(im_fft_magnitude)
    threshold_value = threshold * max_magnitude
    # Create a mask to keep frequencies with magnitude lower than the threshold
    mask = im_fft_magnitude < threshold_value
    # Apply the mask
    im_fft_denoised = im_fft * mask
    # Perform the inverse FFT
    im_new = ifft2(im_fft_denoised).real
    return im_new

def plot_denoised_image(image, denoised_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image, cmap='gray')
    plt.show()

# Compress Image
def generate_compressed_images(image):
    im_fft = fft2(image)
    # Get the magnitude of the FFT
    im_fft_magnitude = np.abs(im_fft)

    compressed_images = []
    percentiles = [0, 50, 75, 90, 95, 99.9]
    for percentile in percentiles:
        # Get the percentile value
        percentile_value = np.percentile(im_fft_magnitude, percentile)
        # Create a mask to identify which values are zeroed out
        mask = im_fft_magnitude > percentile_value
        # Apply the mask to zero out high frequencies
        compressed_image = im_fft * mask
        # Perform the inverse FFT
        im_new = ifft2(compressed_image).real
        # Count the number of non-zero elements in the Fourier transform
        num_non_zeros = np.count_nonzero(mask)
        print(f"Percentile {percentile}: Number of non-zero elements in Fourier transform = {num_non_zeros}")
        compressed_images.append((percentile, im_new))
    return compressed_images

# Plot Compressed Images
def plot_compressed_images(compressed_images):
    plt.figure(figsize=(15, 10))
    for i, x in enumerate(compressed_images):
        plt.subplot(2, 3, i+1)
        plt.title(f"{x[0]}th Percentile")
        plt.imshow(x[1], cmap='gray')
    plt.show()

# Plot Runtime
def plot_runtime():
    sizes = [2**i for i in range(5, 11)]
    dft2_means = []
    fft2_means = []
    dft2_stds = []
    fft2_stds = []
    
    for size in sizes:
        print(f"Running experiment for size {size}x{size}")
        dft2_time_samples = []
        fft2_time_samples = []
        for _ in range(10):  # Run the experiment 10 times
            x = np.random.random((size, size))
            start_time = time.time()
            dft2(x)
            dft2_time_samples.append(time.time() - start_time)
            
            start_time = time.time()
            fft2(x)
            fft2_time_samples.append(time.time() - start_time)
        
        dft2_means.append(np.mean(dft2_time_samples))
        fft2_means.append(np.mean(fft2_time_samples))
        dft2_stds.append(np.std(dft2_time_samples))
        fft2_stds.append(np.std(fft2_time_samples))
    
    plt.errorbar(sizes, dft2_means, yerr=[2*std for std in dft2_stds], label='2D DFT', capsize=5)
    plt.errorbar(sizes, fft2_means, yerr=[2*std for std in fft2_stds], label='2D FFT', capsize=5)
    plt.xlabel('Input size')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title('Runtime Complexity of 2D DFT and 2D FFT')
    plt.show()

    for size, dft2_mean, fft2_mean, dft2_std, fft2_std in zip(sizes, dft2_means, fft2_means, dft2_stds, fft2_stds):
        print(f"Size: {size}, 2D DFT Mean: {dft2_mean:.6f}s, 2D DFT Std: {dft2_std:.6f}s, 2D FFT Mean: {fft2_mean:.6f}s, 2D FFT Std: {fft2_std:.6f}s")

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
        denoised_image = denoise_image(image)
        plot_denoised_image(image[:original_shape[0], :original_shape[1]], denoised_image[:original_shape[0], :original_shape[1]])
    elif args.mode == 3:
        compressed_images = generate_compressed_images(image)
        compressed_images = [(x[0], x[1][:original_shape[0], :original_shape[1]]) for x in compressed_images]
        for percentile, img in compressed_images:
            num_non_zeros = np.count_nonzero(img)
            print(f"Percentile {percentile}: Number of non-zero elements = {num_non_zeros}")
        plot_compressed_images(compressed_images)
    elif args.mode == 4:
        plot_runtime()

if __name__ == '__main__':
    main()
