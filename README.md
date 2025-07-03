# COLOR RECOGNITION

This project focuses on color classifying by K-Nearest Neighbors Machine Learning Classifier which is trained by R, G, B Color Histogram. It can classify White, Black, Red, Green, Blue, Orange, Yellow and Violet. If you want to classify more color or improve the accuracy you should work on the [training data](https://github.com/ahmetozlu/color_classifier/tree/master/src/training_dataset) or consider about other color features such as [Color Moments](https://en.wikipedia.org/wiki/Color_moments) or [Color Correlogram](http://www.cs.cornell.edu/rdz/Papers/ecdl2/spatial.htm).

You can use [color_recognition_api](https://github.com/ahmetozlu/color_recognition/tree/master/src/color_recognition_api) to perform real-time color recognition in your projects. You can find a sample usage of [color_recognition_api](https://github.com/ahmetozlu/color_recognition/tree/master/src/color_recognition_api) in this [**repo**](https://github.com/ahmetozlu/vehicle_counting_tensorflow). ***Please contact if you need professional color recognition project with the super high accuracy!***

## Quick Demo

***Run [color_classification_webcam.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_webcam.py) to perform real-time color recognition on a webcam stream.***

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34917659-8497acae-f95a-11e7-93fb-f7cd6cc3128a.gif">
</p>

***Run [color_classification_image.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_image.py) to perform color recognition on a single image.***

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42423806-14cdfa7a-8309-11e8-9478-23d50fc0002f.png">
</p>

---
**What does this program do?**
1.  **Feature Extraction:** Performs feature extraction to obtain the R, G, B Color Histogram values from training images.
2.  **Training K-Nearest Neighbors Classifier:** Trains a K-Nearest Neighbors (KNN) classifier using the extracted R, G, B Color Histogram values.
3.  **Real-time Classification:** Reads webcam frames or loads a single image, performs feature extraction on the input, and then classifies the dominant color using the trained KNN classifier. The classification result, along with a confidence indicator (votes from K neighbors) and a visual Region of Interest (ROI), is displayed on the screen.
---

## How to Run

**1. Prerequisites:**
*   Python 3.x
*   OpenCV (`pip install opencv-python`)
*   NumPy (`pip install numpy`)
*   Matplotlib (`pip install matplotlib`) - *Optional, for histogram visualization function*

**2. Setup:**
*   Ensure you have the `training_dataset` directory in the same parent directory as your Python scripts, containing subfolders for each color (e.g., `red/`, `blue/`, etc.) with respective training images.
*   The `training.data` and `test.data` files will be created automatically.

**3. Running the Application:**

*   **For Image Classification:**
    ```bash
    python MultipleFiles/color_classification_image.py [path_to_your_image.jpg]
    ```
    If no image path is provided, it defaults to `black_cat.jpg`.

*   **For Webcam Real-time Classification:**
    ```bash
    python MultipleFiles/color_classification_webcam.py
    ```
    Press 'q' to quit the webcam stream.

## Theory

In this study, colors are classified by using K-Nearest Neighbor Machine Learning classifier algorithm. This classifier is trained by image R, G, B Color Histogram values. The general work flow is given at the below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/35335133-a9632c70-0125-11e8-9204-0b4bfd0702a7.png" {width=35px height=350px}>
</p>

You should know 2 main phenomena to understand basic Object Detection/Recognition Systems of Computer Vision and Machine Learning.

**1.) Feature Extraction**

How to represent the interesting points we found to compare them with other interesting points (features) in the image.

**2.) Classification**

An algorithm that implements classification, especially in a concrete implementation, is known as a classifier. The term "classifier" sometimes also refers to the mathematical function, implemented by a classification algorithm, that maps input data to a category.

For this project;

**1.) Feature Extraction** = Color Histogram

Color Histogram is a representation of the distribution of colors in an image. For digital images, a color histogram represents the number of pixels that have colors in each of a fixed list of color ranges, that span the image's color space, the set of all possible colors.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34918867-44f5feaa-f96b-11e7-9994-1747846266c9.png">
</p>

**2.) Classification** = K-Nearest Neighbors Algorithm

K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34918895-c7b94d24-f96b-11e7-87da-8619d9bd4246.png">
</p>

## Implementation

[OpenCV](https://pypi.python.org/pypi/opencv-python) was used for color histogram calculations and knn classifier. [NumPy](https://stackoverflow.com/questions/29499815/how-to-install-numpy-on-windows-using-pip-install) was used for matrix/n-dimensional array calculations. The program was developed on Python at Linux environment.

In the “[src](https://github.com/ahmetozlu/color_recognition/tree/master/src)” folder, there are 2 Python classes which are:

-   **[color_classification_webcam.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_webcam.py):** test class to perform real-time color recognition form webcam stream.

-   **[color_classification_image.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_image.py):** test class to perform color recognition on a single image.

In the “[color_recognition_api](https://github.com/ahmetozlu/color_recognition/tree/master/src/color_recognition_api)” folder, there are 2 Python classes which are:

-   **[feature_extraction.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/color_histogram_feature_extraction.py):** feature extraction operation class

-   **[knn_classifier.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/knn_classifier.py):** knn classification class

**1.) Explanation of “[feature_extraction.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/color_histogram_feature_extraction.py)"**

I can get the RGB color histogram of images by this Python class. For example, plot of RGB color histogram for one of the red images is given at the below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34919478-f198beb8-f975-11e7-8c1c-0a552f7cd673.jpg" {width=25px height=250px}>
</p>

I decided to use bin number of histogram which has the peak value of pixel count for R, G and B as feature so I can get the dominant R, G and B values to create feature vectors for training. For example, the dominant R, G and B values of the red image which is given at above is [254, 0, 2].

I get the dominant R, G, B values by using Color Histogram for each training image then I labelled them because KNN classifier is a supervised learner and I deploy these feature vectors in the csv file. Thus, I create my training feature vector dataset. It can be found in the file which name’s is [training.data](https://github.com/ahmetozlu/color_recognition/blob/master/src/training.data) under src folder.

**2.) Explanation of “[knn_classifier.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/knn_classifier.py)”**

This class provides these main calculations;

1.  Fetching training data
2.  Fetching test image features
3.  Calculating euclidean distance
4.  Getting k nearest neighbors
5.  Prediction of color
6.  Returning the prediction and its confidence (votes from neighbors)

**“[color_classification_webcam.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_webcam.py)”** is the main class of my program, it provides;

1.  Calling [feature_extraction.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/color_histogram_feature_extraction.py) to create training data by feature extraction
2.  Calling [knn_classifier.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/knn_classifier.py) for classification
3.  Enhanced visual feedback including ROI, prediction, confidence, and FPS.

You can find training data in [here](https://github.com/ahmetozlu/color_classifier/tree/master/src/training_dataset).

You can find features are got from training data in [here](https://raw.githubusercontent.com/ahmetozlu/color_classifier/master/src/training.data).

## Conclusion

I think, the training data has a huge important in classification accuracy. I created my training data carefully but maybe the accuracy can be higher with more suitable training data.

Another important thing is lightning and shadows. In my test images, the images which were taken under bad lighting conditions and with shadows are classified wrong (false positives), maybe some filtering algorithm should/can be implemented before the test images send to KNN classifier Thus, accuracy can be improved.

## Future Enhancements (TODOs)

*   **"Add New Color" Utility:** Implement a user-friendly utility to add new colors to the training dataset and re-train the classifier.
*   **Advanced Feature Extractors:** Integrate and allow selection of other color features like Color Moments or Color Correlogram for potentially higher accuracy.
*   **Alternative Classifiers:** Incorporate other machine learning algorithms (e.g., SVM, Random Forest) for comparison and improved performance.
*   **GUI Application:** Develop a full graphical user interface for easier interaction, parameter tuning, and visualization.
*   **Robustness to Lighting:** Explore pre-processing techniques (e.g., illumination normalization) to make the classifier more robust to varying lighting conditions and shadows.
*   **Performance Profiling:** Optimize the code for better real-time performance, especially for higher resolution streams.

## Citation
If you use this code for your publications, please cite it as:

    @ONLINE{cr,
        author = RAM
        title   = "Color Recognition",
        year   = "2025",
        url    = "https://github.com/ramthedevhub"
    }

## Author
Ahmet Özlü

## License
This system is available under the MIT license. See the LICENSE file for more info.
