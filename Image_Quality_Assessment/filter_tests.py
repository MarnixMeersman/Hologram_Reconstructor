import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

def load_image_from_npy(npy_file, index):
    images = np.load(npy_file)
    return images[index]

def apply_edge_detection(image):
    # Using Sobel filter for edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return np.hypot(sobelx, sobely)

def apply_frequency_domain_analysis(image):
    # Frequency domain transformation
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return magnitude_spectrum

def apply_variance_of_laplacian(image):
    # Variance of Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

def plot_results(original, edge_detection, frequency_domain, variance_laplacian):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(edge_detection, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(frequency_domain, cmap='gray')
    plt.title('Frequency Domain Analysis')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f'Variance of Laplacian: {variance_laplacian}',
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.title('Variance of Laplacian')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    npy_file = '/reconstructions/20231212-140735/reconstructed_holograms.npy'  # Replace with your .npy file path
    image_index = 1  # Index of the image you want to analyze

    image = load_image_from_npy(npy_file, image_index)
    if len(image.shape) == 3 and image.shape[2] == 3:  # Convert color images to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge_detection_result = apply_edge_detection(image)
    frequency_domain_result = apply_frequency_domain_analysis(image)
    variance_laplacian_result = apply_variance_of_laplacian(image)

    plot_results(image, edge_detection_result, frequency_domain_result, variance_laplacian_result)
