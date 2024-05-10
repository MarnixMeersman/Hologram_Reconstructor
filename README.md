# Hologram Reconstructor

This project is a Python application that reconstructs holograms. It uses multiprocessing to speed up the reconstruction process and provides options for edge detection and blob detection. The reconstructed images can be visualized with a slider or saved as separate files.

## Features

- **Multiprocessing**: The application uses multiprocessing to speed up the reconstruction process.
- **Edge Detection**: The application provides an option to apply edge detection to the reconstructed images.
- **Blob Detection**: The application provides an option to apply blob detection to the reconstructed images.
- **Image Visualization**: The reconstructed images can be visualized with a slider.
- **Image Saving**: The reconstructed images can be saved as separate files.

![Uploading ScreenRecording2024-05-10at23.59.39-ezgif.com-video-to-gif-converter.gif…]()


## Usage

To use the application, follow these steps:

1. Clone the repository and install the necessary libraries as described in the "Getting Started" section.
2. The `main.py` script in your project accepts several command-line arguments that control its behavior. Here are the options:

- `--apply_edges`: If this option is provided, the script will apply edge detection to the reconstructed images.
- `--visualize`: If this option is provided, the script will visualize the reconstructed images with a slider.
- `--apply_blob_detection`: If this option is provided, the script will apply blob detection to the reconstructed images.
- `--save_as_separate`: If this option is provided, the script will save each reconstructed image as a separate file.

You can use these options by appending them to the `python main.py` command. For example, if you want to apply edge detection and visualize the results with a slider, you can run:

```bash
python main.py --apply_edges --visualize
```

If you want to apply blob detection and save each image separately, you can run:

```bash
python main.py --apply_blob_detection --save_as_separate
```

More features -- like reconstruction heights, number of reconstructions, and the optical assembly parameters can be configured in `main.py`.
These need to be changed in the code when you make your optical assembly, but remain constant as long as the setup is not changed.



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires Python 3.9 or higher and pip installed on your machine.

### Cloning the Repository

To clone the repository, open a terminal and run the following command:

```bash
git clone git@github.com:/MarnixMeersman/Hologram_Reconstructor.git
```
### Installing the Required Libraries
Navigate to the project directory and run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```
