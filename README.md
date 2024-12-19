# FFT Image Processing Project

This project demonstrates the application of Fourier Transform algorithms, including the **Discrete Fourier Transform (DFT)** and **Fast Fourier Transform (FFT)**, for processing images. It includes implementations of 1D and 2D DFT/FFT, image denoising, compression, and runtime analysis.

## Features

- **Naive DFT/IDFT**: A simple implementation of the Discrete Fourier Transform.
- **Cooley-Tukey FFT/IFFT**: Optimized implementation of the Fast Fourier Transform.
- **2D DFT/FFT and IDFT/IFFT**: Extensions for 2D image processing.
- **Image Denoising**:
  - Retain low-frequency components for noise reduction.
  - Threshold-based filtering of high-frequency noise.
- **Image Compression**:
  - Visualize compression by retaining frequency components above specified percentiles.
- **Runtime Analysis**: Compare the computational efficiency of 2D DFT and FFT.
- **Image Padding**: Ensure compatibility with FFT by padding images to the nearest power of 2.

## Requirements

- Python 3.7 or higher.
- Required libraries: `numpy`, `matplotlib`, `opencv-python`.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main program using the command line:

```bash
python fft_image_processing.py -m MODE -i IMAGE
```

### Arguments

- `-m`, `--mode`: Operation mode (1 for FFT visualization, 2 for denoising, 3 for compression, 4 for runtime analysis).
- `-i`, `--image`: Path to the image file (default: `moonlanding.png`).

### Modes

1. **FFT Visualization**:
   - Displays the original image and its FFT magnitude spectrum.
   - Example:
     ```bash
     python fft_image_processing.py -m 1 -i moonlanding.png
     ```

2. **Image Denoising**:
   - Removes high-frequency noise while retaining low-frequency information.
   - Example:
     ```bash
     python fft_image_processing.py -m 2 -i moonlanding.png
     ```

3. **Image Compression**:
   - Compresses the image by retaining only high-magnitude frequencies.
   - Displays results for multiple percentile thresholds.
   - Example:
     ```bash
     python fft_image_processing.py -m 3 -i moonlanding.png
     ```

4. **Runtime Analysis**:
   - Measures and compares the runtimes of 2D DFT and FFT for various input sizes.
   - Example:
     ```bash
     python fft_image_processing.py -m 4
     ```

## Key Functions

- **Fourier Transform Implementations**:
  - `naive_dft`, `naive_idft`: Basic DFT and IDFT implementations.
  - `fft`, `ifft`: Recursive Cooley-Tukey FFT and IFFT implementations.
  - `dft2`, `fft2`, `ifft2`: 2D extensions for image processing.

- **Denoising**:
  - `denoise_image`: Retain low-frequency components for noise reduction.
  - `denoise_image_threshold`: Use a magnitude threshold for noise filtering.

- **Compression**:
  - `generate_compressed_images`: Retain frequency components above specified percentiles.
  - `plot_compressed_images`: Visualize compressed images and non-zero frequency counts.

- **Runtime Analysis**:
  - `plot_runtime`: Measure and compare runtimes of DFT and FFT for various input sizes.

## Outputs

- **Visualizations**:
  - Original image alongside its FFT magnitude spectrum.
  - Denoised and compressed images.
  - Runtime comparison graph.

- **Console Logs**:
  - Number and fraction of retained frequency components in compression and denoising.
  - Runtime statistics for DFT and FFT.

## Example Images

1. **FFT Visualization**:
   - Shows the original image and the magnitude of its FFT.

2. **Denoising**:
   - Displays the original and denoised images.

3. **Compression**:
   - Compressed images with retained frequency components for different thresholds.

4. **Runtime Analysis**:
   - Graph comparing the execution time of DFT and FFT for various input sizes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, feel free to open an issue or submit a pull request.

---

Happy processing! 🎨✨
Python version used: Python 3.13.0
