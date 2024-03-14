import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy.fft import fft2, ifft2, fftshift

def load_hologram(file_path):
    return imread(file_path, mode='F')

def reconstruct_hologram(hologram, wavelength, distance, pixel_size):
    Ny, Nx = hologram.shape
    k = 2 * np.pi / wavelength
    fx = np.linspace(-Nx / 2, Nx / 2, Nx) / (Nx * pixel_size)
    fy = np.linspace(-Ny / 2, Ny / 2, Ny) / (Ny * pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * distance) * np.exp(-1j * np.pi * wavelength * distance * (FX ** 2 + FY ** 2))
    H = fftshift(H)
    U1 = fft2(hologram)
    U2 = H * U1
    U3 = ifft2(U2)
    return np.abs(U3), U1, U2

def display_image(data, title, is_complex=False):
    if is_complex:
        data = np.abs(data)
    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def main(hologram_path, distance, pixel_size, wavelength):
    hologram = load_hologram(hologram_path)
    reconstructed_image, U1, U2 = reconstruct_hologram(hologram, wavelength, distance, pixel_size)

    display_image(reconstructed_image, 'Reconstructed Image')
    display_image(np.abs(U1), 'U1', is_complex=True)
    display_image(np.abs(U2), 'U2', is_complex=True)

if __name__ == '__main__':
    wavelength = 650e-9  # example wavelength
    pixel_size = 1.12e-6 # example pixel size
    distance = 0.01     # example distance

    hologram_path = 'holograms/saliva.jpg' # Replace with your hologram image path
    main(hologram_path, distance, pixel_size, wavelength)
