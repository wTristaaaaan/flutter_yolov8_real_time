# Training and Implementing YOLOv8 with Flutter

This project aims to develop a Flutter application capable of real-time object detection using the YOLOv8 model. We will outline the steps for training the model and integrating it into a Flutter application.

## Prerequisites

//

## Step 1: Selecting and Preparing the Dataset

### Finding a Dataset

To train our YOLOv8 model, we need a substantial dataset of human faces. We have selected a dataset available on Kaggle at the following link: [Human Faces Object Detection Dataset](https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection?resource=download). This dataset includes annotated images ready for object detection model training.

### Preparing and Storing the Dataset

Once the dataset is downloaded:

1. Unzip the archive and check the format of the data (usually images and an annotation file in XML or JSON).
2. Upload the uncompressed data to your Google Drive. Make sure to note the path to the folder containing the data, as it will be needed for the training step.

## Step 2: Setting Up the Development Environment

### Configuring Google Colab

To set up your development environment for training the YOLOv8 model, follow these steps:

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Once you are in your Colab notebook, click on `Runtime` in the top menu.
3. Select `Change runtime type` from the dropdown menu.
4. In the dialog that appears, set the `Hardware accelerator` to `TPUv2` to leverage Google's Tensor Processing Units, which are optimized for machine learning tasks and will accelerate the training process.
5. Click `Save` to apply the changes.

## Step 3: Data Augmentation

In order to enhance the robustness of our YOLOv8 model and to improve its ability to generalize from our dataset, we will implement a data augmentation strategy. This process involves generating new training images by applying various transformations to the existing images in our dataset. This helps the model perform better in diverse lighting and environmental conditions.

### Augmentation Techniques

We will use the following techniques to create five augmented versions of each image:

1. **Brightness Adjustment**: Modifying the brightness of the image to simulate different lighting conditions.
2. **Contrast Adjustment**: Altering the contrast to make the images appear either more vivid or subdued.
3. **Sharpness Enhancement**: Increasing the sharpness to highlight details or decreasing it to simulate a softer focus.
4. **Shadow Adjustment**: Adjusting the shadows to simulate different times of the day or various light sources.
5. **Color Modification**: Changing the color balance and saturation to test the model's color sensitivity and accuracy.

### Implementation

The augmentation will be performed using a script that takes an input image and applies the aforementioned transformations. Each transformation will result in a new image, thus for each original image, five new images will be generated.

Here is a brief overview of how the script will work:

- The script reads an original image from the dataset.
- It applies each of the five transformations sequentially, saving each new image to the dataset.
- The original image and its augmented versions are then used for training the YOLOv8 model.

This method of augmentation not only diversifies the training data but also simulates real-world scenarios where lighting and environmental conditions can vary significantly.

### Benefits of Data Augmentation

By augmenting our data, we aim to achieve the following:

- Enhance the model's ability to detect objects under various environmental conditions.
- Improve the generalization capabilities of the model, reducing the likelihood of overfitting.
- Increase the dataset size, which is crucial for training deep learning models effectively.

This approach ensures that our model is well-trained and robust, capable of performing high-quality object detection in different settings.

## Note on Processing Delays in Google Colab

When working with Google Colab to manipulate files, especially when creating new images and directories directly in Google Drive, users may experience significant processing delays. This section provides guidance on managing these delays and troubleshooting potential issues.

### Understanding Processing Delays

Creating and saving large numbers of images or extensive directory structures in Google Drive from Google Colab can be time-consuming. This is due to the interactions between the Colab environment and the Google Drive API, which can sometimes be slow, particularly:

- **Network Latency**: Since files are being saved to the cloud, operations depend on network speed and stability.
- **API Throttling**: Google Drive API might limit the rate of requests, which can slow down file creation and retrieval.
- **High Volume Operations**: Processing and uploading large sets of images involves considerable data transfer, which naturally takes time.

### Best Practices for Managing Delays

To effectively manage these delays and ensure a smooth workflow, consider the following tips:

- **Batch Operations**: Where possible, batch operations together to minimize the number of individual read/write requests to the Drive.
- **Asynchronous Checks**: After triggering file creation operations, periodically check for their completion in an asynchronous manner rather than waiting synchronously.
- **Error Handling**: Implement robust error handling to retry operations that fail due to timeouts or API rate limits.

### Troubleshooting Errors

If you encounter errors or do not see your files and directories appearing in Google Drive as expected, take the following steps:

1. **Patience is Key**: Sometimes, all that’s required is a bit more time for changes to propagate and files to appear.
2. **Manual Refresh**: Manually refresh your Google Drive folder view to check if the files have been created but not yet displayed.
3. **Check Logs**: Review the output and error logs in Google Colab for any indications of what might have gone wrong.
4. **Restart the Runtime**: If persistent issues occur, try restarting the Colab runtime to clear any potential glitches.

### Conclusion

While working with files directly in Google Drive via Google Colab is incredibly powerful, it is important to account for potential delays and plan for handling errors. With the right approach, you can minimize disruption and maintain productivity.

## Step 4: Training the YOLOv8 Model

In this step, we will focus on training the YOLOv8 model using the augmented data prepared in the previous steps. Considering our application's need for real-time performance, we have chosen to use the YOLOv8n model. This decision is based on its optimal balance of speed and accuracy, making it highly suitable for real-time object detection in mobile applications.

### Choosing the Model Configuration

Based on the performance metrics provided in the official [Ultralytics documentation](https://docs.ultralytics.com/fr/tasks/detect/#models), we have selected the YOLOv8n model for our project. Here's why:

| Model   | Size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | Params (M) | FLOPs (B) |
| ------- | ------------- | ------------ | ------------------- | ------------------------ | ---------- | --------- |
| YOLOv8n | 640           | 37.3         | 80.4                | 0.99                     | 3.2        | 8.7       |

As illustrated, YOLOv8n provides a substantial 37.3 mAP with an incredibly fast inference time of just 0.99 ms on an A100 using TensorRT, which is crucial for achieving real-time performance on mobile devices.

### Training Process

1. **Setting Up the Environment**: Ensure that your Google Colab runtime is set up with a TPU or GPU acceleration to facilitate efficient model training.
2. **Loading the Data**: Load the augmented dataset from Google Drive, ensuring the data paths are correctly configured.
3. **Configuring the Training Parameters**: Set up the training parameters, including learning rate, batch size, and number of epochs. For YOLOv8n, consider starting with the default settings recommended by Ultralytics and adjust based on preliminary results.
4. **Launching the Training**: Use the Ultralytics YOLOv8 training scripts, which are optimized for various hardware configurations. Ensure to monitor the training progress through logs and TensorBoard to keep track of loss metrics and validation accuracy.
5. **Model Evaluation**: After training, evaluate the model using the validation set to check the accuracy and make any necessary adjustments to the training regimen.

### Implementation Details

- **Script Execution**: All training scripts and commands should be executed within the Google Colab environment, utilizing its powerful hardware capabilities.
- **Model Saving**: Regularly save checkpoints to Google Drive to prevent data loss and allow training to resume from the last checkpoint in case of interruptions.

### Real-Time Implementation Considerations

Given the YOLOv8n's high performance, special attention should be paid to integration details for real-time application:

- **Optimization**: Further optimize the trained model using tools like TensorRT or ONNX for faster inference speeds.
- **Testing**: Rigorously test the model in diverse real-world scenarios to ensure it maintains high accuracy and performance under different conditions.

### Conclusion

The choice of YOLOv8n aligns with our goal of developing a real-time object detection system in a Flutter application. Its impressive speed and good accuracy offer the best trade-off for our requirements, promising robust performance in a mobile environment.

### Step 4.1: Preparing the Dataset for YOLOv8 Training

When training a YOLOv8 model, it's essential to structure your dataset correctly to ensure that the model trains effectively and efficiently. The dataset must be organized into separate directories for training and validation, each containing subdirectories for images and labels.

#### Dataset Structure

The YOLOv8 model expects the following directory structure for the dataset:

| Directory  | Subdirectory | Description                                                     |
| ---------- | ------------ | --------------------------------------------------------------- |
| `dataset/` |              | Root folder for dataset                                         |
|            | `train/`     | Contains training data                                          |
|            |              | - `images/`: Contains all training images                       |
|            |              | - `labels/`: Contains corresponding .txt files with annotations |
|            | `val/`       | Contains validation data                                        |
|            |              | - `images/`: Contains all validation images                     |
|            |              | - `labels/`: Contains corresponding .txt files with annotations |

#### Image and Label Correspondence

Each image in the `images/` subdirectory should have a corresponding `.txt` file in the `labels/` subdirectory. The names of the files in these subdirectories should match, indicating their correspondence. For example, if you have an image named `train_image1.jpg`, its corresponding label should be named `train_image1.txt`.

#### Label File Format

Each `.txt` file contains annotations for the objects detected in the corresponding image. The format for each line in the label files is as follows:

<class_id> <x_center> <y_center> width height

Where:

- `<class_id>` is an integer representing the object class.
- `<x_center>` and `<y_center>` are the normalized x and y coordinates of the center of the bounding box, relative to the dimensions of the image.
- `<width>` and `<height>` are the normalized width and height of the bounding box, relative to the dimensions of the image.

#### Importance of Data Format

Ensuring that the images and labels are correctly formatted and correspond to each other is crucial for successful model training. The YOLOv8 model uses these annotations to learn the characteristics of different objects, and any discrepancy between images and labels could lead to poor model performance or training errors.

This structured approach not only helps in maintaining an organized dataset but also streamlines the process of model training by ensuring that the training script can efficiently locate and use the necessary files for both training and validation phases.

### Step 5: Training the YOLOv8 Model with Ultralytics Library

After preparing our dataset and setting up the development environment, we proceeded to train the YOLOv8 model using the Ultralytics YOLO library. This library provides a streamlined interface for training YOLO models with custom datasets.

#### Training Code Explanation

We used the following Python code to initiate the training of our model:

```python
from yolov8 import YOLO

# Initialize the model with the specified weights
model = YOLO('yolov8n.pt')

# Start the training process
results = model.train(
    data='/content/drive/MyDrive/YOLOv8/NewDataset/yolo.yml',
    imgsz=320,  # Size of the images
    epochs=10,  # Total number of training epochs
    batch=16,   # Batch size
    name='faces_detection',  # Name of the training run for checkpoint saving
    cache=True  # Enables caching images for faster loading
)
```
