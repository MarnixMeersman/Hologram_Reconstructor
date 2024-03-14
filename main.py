"""
This script is used to reconstruct a hologram from a digital hologram image. The hologram is reconstructed at multiple
heights.

Interaction with the script is as follows:
scroll down to the main function and set the parameters as desired. The parameters are:
- hologram_path: the path to the hologram image
- distances: a list of distances at which the hologram should be reconstructed
- pixel_size: the size of the pixels in the hologram image
- wavelength: the wavelength of the light used to record the hologram
- apply_edges: a boolean to indicate if edge detection should be applied to the reconstructed holograms
- blob_detection: a boolean to indicate if blob detection should be applied to the edge-detected holograms
- slider: a boolean to indicate if a slider should be shown to visualize the reconstructed holograms
- save_as_separate: a boolean to indicate if the reconstructed holograms should be saved as separate images (takes long)
    only recommended when happy with the results you see in the slider function

The script uses multiprocessing to speed up the reconstruction process. The number of CPU cores is automatically
determined and used to create a pool of workers.


Author: Marnix F.L. Meersman
Email: m.f.l.meersman@student.tudelft.nl
Date: 05-12-2024
"""

# Importing the necessary libraries
import numpy as np
import os
import cv2
import multiprocessing
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm
from imageio import imread, imwrite
import datetime
from mpl_toolkits.mplot3d import Axes3D


# Functions
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
def reconstruct_at_height(args):
    hologram, wavelength, height, pixel_size = args
    print(f"Reconstructing at height: {height*1000} mm...")
    return reconstruct_hologram(hologram, wavelength, height, pixel_size)[0]
def apply_edge_detection(image):
    # Using Sobel filter for edge detection on the original image
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=31)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=31)

    # Calculate the magnitude of gradients
    edges = np.hypot(sobelx, sobely)

    # Normalize the edge data to 0-1 range
    norm_edges = edges / np.max(edges)

    return norm_edges
def detect_blobs(image, sharpness_threshold=50):
    # Ensure image is 8-bit
    if image.dtype != np.uint8:
        image = np.uint8(255 * image)  # Assuming image is normalized to 0-1 range

    # Set up the detector with customized parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 500
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = False

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)

    # # Debugging: Print number of keypoints
    # print(f"Number of keypoints detected: {len(keypoints)}")

    # Draw detected blobs as red circles
    blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return blobs, keypoints
def plot_blobs_in_3d(blob_coordinates, distances):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for coords, z in zip(blob_coordinates, distances):
        # Extract x and y coordinates from each KeyPoint
        xs = [kp.pt[0] for kp in coords]
        ys = [kp.pt[1] for kp in coords]
        zs = [z] * len(coords)

        ax.scatter(xs, ys, zs)

    ax.set_xlabel('1.12 micron pixels')
    ax.set_ylabel('1.12 micron pixels')
    ax.set_zlabel('z-axis [m]')
    plt.show()
def visualize_images_with_slider(image_array):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Display the first image
    img_display = plt.imshow(image_array[0], cmap='gray')
    ax.margins(x=0)

    # Make a horizontal slider to control the current image
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Image Index', 0, len(image_array) - 1, valinit=0, valfmt='%0.0f')

    # Update the image index on slider change
    def update(val):
        img_display.set_data(image_array[int(slider.val)])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def save_images_separately(images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Get the shape of the first image
    height, width = images[0].shape

    for idx, img in enumerate(images):
        plt.figure(figsize=(width/80, height/80), dpi=80)  # 80 is the default DPI in matplotlib
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(folder_path, f'image_{idx}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

############################################################################################################
# Main function
def main(hologram_path, distances, pixel_size, wavelength,
         apply_edges=False, visualize=False, apply_blob_detection=False):
    hologram = load_hologram(hologram_path)

    cpu_count = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {cpu_count}")
    pool = multiprocessing.Pool(processes=cpu_count)
    print(f"Multiprocessing pool created with {cpu_count} workers.")

    tasks = [(hologram, wavelength, d, pixel_size) for d in distances]
    print(f"Starting reconstruction for {len(distances)} heights...")

    results = pool.map(reconstruct_at_height, tasks)

    pool.close()
    pool.join()
    print("All reconstructions completed.")

    if apply_edges:
        print("Applying edge detection to reconstructed images...")
        results = [apply_edge_detection(np.float32(img)) for img in results]

    blob_coords_per_height = []
    if apply_blob_detection:
        print("Applying blob detection to edge-detected images...")
        results = [detect_blobs(img)[0] for img in results]

        for idx, img in enumerate(results):
            blobs = detect_blobs(img)[1]
            blob_coords_per_height.append(blobs)





    reconstructions_dir = 'reconstructions'
    if not os.path.exists(reconstructions_dir):
        os.makedirs(reconstructions_dir)

    np.save(os.path.join(reconstructions_dir, 'reconstructed_holograms.npy'), np.array(results))
    print("All reconstructed holograms saved in 'reconstructions' folder.")

    if visualize:
        visualize_images_with_slider(results)
    if apply_blob_detection:
        plot_blobs_in_3d(blob_coords_per_height, distances)

    # Determine the folder name based on current date and time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    reconstructions_dir = os.path.join('reconstructions', current_time)
    if not os.path.exists(reconstructions_dir):
        os.makedirs(reconstructions_dir)

    if save_as_separate:
        print("Saving reconstructed images as separate files...")
        save_images_separately(results, reconstructions_dir)
    else:
        # Save all reconstructed images in a single .npy file
        np.save(os.path.join(reconstructions_dir, 'reconstructed_holograms.npy'), np.array(results))
        print("All reconstructed holograms saved in 'reconstructions' folder.")



if __name__ == '__main__':
    apply_edges = False
    blob_detection = False
    slider      = True
    save_as_separate = False

    wavelength  = 650e-9 # meters
    pixel_size  = 1.12e-6 # meters
    distances   = list(np.linspace(0.005e-4, 1.3e-2, 30))

    hologram_path = 'holograms/saliva.jpg'
    print("Holographic reconstruction script started.")
    main(hologram_path, distances, pixel_size, wavelength, apply_edges=apply_edges, visualize=slider,
         apply_blob_detection=blob_detection)
    print("\n\nHolographic reconstruction script completed.")
