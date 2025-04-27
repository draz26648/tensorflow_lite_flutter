# TensorFlow Lite Flutter

[![pub package](https://img.shields.io/pub/v/tensorflow_lite_flutter.svg)](https://pub.dev/packages/tensorflow_lite_flutter)

A comprehensive Flutter plugin for accessing TensorFlow Lite API. This plugin provides a Dart interface to TensorFlow Lite models, allowing Flutter apps to perform on-device machine learning with high performance and low latency.

## Features

Supports multiple ML tasks on both iOS and Android:

- ✅ Image Classification
- ✅ Object Detection (SSD MobileNet and YOLO)
- ✅ Pix2Pix Image-to-Image Translation
- ✅ Semantic Segmentation (Deeplab)
- ✅ Pose Estimation (PoseNet)

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
  - [Android Configuration](#android-configuration)
  - [iOS Configuration](#ios-configuration)
- [Usage](#usage)
  - [Loading Models](#loading-models)
  - [Image Classification](#image-classification)
  - [Object Detection](#object-detection)
  - [Pix2Pix](#pix2pix)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Pose Estimation](#pose-estimation)
- [Advanced Usage](#advanced-usage)
  - [GPU Acceleration](#gpu-acceleration)
  - [Performance Optimization](#performance-optimization)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Version History

### v3.0.0 (Latest)

- Updated to support Flutter 3.16+ and Dart 3.2+
- Improved documentation and examples
- Performance optimizations

### v2.0.1

- iOS TensorFlow Lite library upgraded to TensorFlowLiteObjC 2.x
- Changes to native code are denoted with `TFLITE2`

### v1.0.0

- Updated to TensorFlow Lite API v1.12.0
- No longer accepts parameter `inputSize` and `numChannels` (retrieved from input tensor)
- `numThreads` moved to `Tflite.loadModel`

## Installation

Add `tensorflow_lite_flutter` as a dependency in your `pubspec.yaml` file:

```yaml
dependencies:
  flutter:
    sdk: flutter
  tensorflow_lite_flutter: ^3.0.0
```

Then run:

```bash
flutter pub get
```

## Setup

### Android Configuration

1. In `android/app/build.gradle`, add the following setting in the `android` block to ensure TensorFlow Lite model files aren't compressed:

```gradle
aaptOptions {
    noCompress 'tflite'
    noCompress 'lite'
}
```

2. If you're using models larger than 100MB, you may need to enable split APKs by adding the following to your `android/app/build.gradle` file:

```gradle
android {
    // Other settings...
    defaultConfig {
        // Other settings...
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
    
    splits {
        abi {
            enable true
            reset()
            include 'armeabi-v7a', 'arm64-v8a'
            universalApk false
        }
    }
}
```

### iOS Configuration

Solutions to common build errors on iOS:

1. **'vector' file not found**

   Open `ios/Runner.xcworkspace` in Xcode, click Runner > Targets > Runner > Build Settings, search for `Compile Sources As`, and change the value to `Objective-C++`

2. **'tensorflow/lite/kernels/register.h' file not found**

   The plugin assumes the TensorFlow header files are located in path "tensorflow/lite/kernels".
   
   For earlier versions of TensorFlow, the header path may be "tensorflow/contrib/lite/kernels".
   
   Use `CONTRIB_PATH` to toggle the path. Uncomment `//#define CONTRIB_PATH` in the iOS implementation if needed.

3. **Memory usage issues**

   For large models, you may need to increase the memory available to your app. Add the following to your `ios/Runner/Info.plist`:

   ```xml
   <key>NSAppTransportSecurity</key>
   <dict>
       <key>NSAllowsArbitraryLoads</key>
       <true/>
   </dict>
   ```

## Usage

### Getting Started

1. Create an `assets` folder and place your model and label files in it. Add them to your `pubspec.yaml`:

```yaml
assets:
  - assets/labels.txt
  - assets/mobilenet_v1_1.0_224.tflite
```

2. Import the library in your Dart code:

```dart
import 'package:tensorflow_lite_flutter/tensorflow_lite_flutter.dart';
```

### Loading Models

Before using any TensorFlow Lite model, you need to load it into memory:

```dart
Future<void> loadModel() async {
  try {
    String? result = await Tflite.loadModel(
      model: "assets/mobilenet_v1_1.0_224.tflite",
      labels: "assets/labels.txt",
      numThreads: 2,         // Number of threads to use (default: 1)
      isAsset: true,         // Is the model file an asset or a file? (default: true)
      useGpuDelegate: false  // Use GPU acceleration? (default: false)
    );
    print('Model loaded successfully: $result');
  } catch (e) {
    print('Failed to load model: $e');
  }
}
```

### Releasing Resources

When you're done using the model, release the resources to free up memory:

```dart
Future<void> disposeModel() async {
  await Tflite.close();
  print('Model resources released');
}
```

### GPU Acceleration

To use GPU acceleration for faster inference:

1. Set `useGpuDelegate: true` when loading the model
2. For optimal performance in release mode, follow the [TensorFlow Lite GPU delegate optimization guide](https://www.tensorflow.org/lite/performance/gpu#step_5_release_mode)

```dart
// Example with GPU acceleration enabled
await Tflite.loadModel(
  model: "assets/model.tflite",
  labels: "assets/labels.txt",
  useGpuDelegate: true  // Enable GPU acceleration
);
```

> **Note**: GPU acceleration works best for floating-point models and may not improve performance for quantized models.

### Image Classification

#### Overview

Image classification identifies what's in an image from a predefined set of categories. This plugin supports various image classification models like MobileNet, EfficientNet, and custom TensorFlow Lite models.

#### Output Format

The model returns a list of classifications with their confidence scores:

```json
[
  {
    "index": 0,
    "label": "person",
    "confidence": 0.629
  },
  {
    "index": 1,
    "label": "dog",
    "confidence": 0.324
  }
]
```

#### Classifying Images

**From a file path:**

```dart
Future<void> classifyImage(String imagePath) async {
  try {
    // Run inference
    List? recognitions = await Tflite.runModelOnImage(
      path: imagePath,       // Required: Path to the image file
      imageMean: 127.5,      // Default: 117.0 (depends on your model)
      imageStd: 127.5,       // Default: 1.0 (depends on your model)
      numResults: 5,         // Default: 5 (maximum number of results)
      threshold: 0.2,        // Default: 0.1 (minimum confidence threshold)
      asynch: true           // Default: true (run in background)
    );
    
    // Process results
    if (recognitions != null) {
      for (var result in recognitions) {
        print('${result["label"]} - ${(result["confidence"] * 100).toStringAsFixed(2)}%');
      }
    }
  } catch (e) {
    print('Error classifying image: $e');
  }
}
```

**From binary data (useful for camera frames):**

```dart
Future<void> classifyImageBinary(Uint8List imageBytes, int inputSize) async {
  try {
    // Process image data to match model input requirements
    Uint8List processedData = imageToByteListFloat32(imageBytes, inputSize, 127.5, 127.5);
    
    // Run inference
    List? recognitions = await Tflite.runModelOnBinary(
      binary: processedData, // Required: Processed image data
      numResults: 5,         // Default: 5
      threshold: 0.1,        // Default: 0.1
      asynch: true           // Default: true
    );
    
    // Process results
    if (recognitions != null) {
      for (var result in recognitions) {
        print('${result["label"]} - ${(result["confidence"] * 100).toStringAsFixed(2)}%');
      }
    }
  } catch (e) {
    print('Error classifying binary image: $e');
  }
}

// Helper function to prepare image data
Uint8List imageToByteListFloat32(Uint8List imageBytes, int inputSize, double mean, double std) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  
  // Process image data to match model input format
  // ... (implementation depends on your image processing needs)
  
  return convertedBytes.buffer.asUint8List();
}

Uint8List imageToByteListFloat32(
    img.Image image, int inputSize, double mean, double std) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      buffer[pixelIndex++] = (img.getRed(pixel) - mean) / std;
      buffer[pixelIndex++] = (img.getGreen(pixel) - mean) / std;
      buffer[pixelIndex++] = (img.getBlue(pixel) - mean) / std;
    }
  }
  return convertedBytes.buffer.asUint8List();
}

Uint8List imageToByteListUint8(img.Image image, int inputSize) {
  var convertedBytes = Uint8List(1 * inputSize * inputSize * 3);
  var buffer = Uint8List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      buffer[pixelIndex++] = img.getRed(pixel);
      buffer[pixelIndex++] = img.getGreen(pixel);
      buffer[pixelIndex++] = img.getBlue(pixel);
    }
  }
  return convertedBytes.buffer.asUint8List();
}
```

- Run on image stream (video frame):

> Works with [camera plugin 4.0.0](https://pub.dartlang.org/packages/camera). Video format: (iOS) kCVPixelFormatType_32BGRA, (Android) YUV_420_888.

```dart
var recognitions = await Tflite.runModelOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height,
  imageWidth: img.width,
  imageMean: 127.5,   // defaults to 127.5
  imageStd: 127.5,    // defaults to 127.5
  rotation: 90,       // defaults to 90, Android only
  numResults: 2,      // defaults to 5
  threshold: 0.1,     // defaults to 0.1
  asynch: true        // defaults to true
);
```

### Object Detection

#### Overview

Object detection identifies and locates objects within an image. This plugin supports two popular object detection architectures:

1. **SSD MobileNet** - Fast and efficient for mobile devices
2. **YOLO** (You Only Look Once) - Higher accuracy but more computationally intensive

#### SSD MobileNet

**Output Format:**

```json
[
  {
    "detectedClass": "hot dog",
    "confidenceInClass": 0.923,
    "rect": {
      "x": 0.15,  // Normalized coordinates (0-1)
      "y": 0.33,  // Normalized coordinates (0-1)
      "w": 0.80,  // Width as percentage of image width
      "h": 0.27   // Height as percentage of image height
    }
  },
  {
    "detectedClass": "person",
    "confidenceInClass": 0.845,
    "rect": {
      "x": 0.52,
      "y": 0.18,
      "w": 0.35,
      "h": 0.75
    }
  }
]
```

**Detecting Objects from an Image File:**

```dart
Future<void> detectObjectsOnImage(String imagePath) async {
  try {
    // Run inference
    List? detections = await Tflite.detectObjectOnImage(
      path: imagePath,         // Required: Path to the image file
      model: "SSDMobileNet",    // Default: "SSDMobileNet"
      imageMean: 127.5,        // Default: 127.5
      imageStd: 127.5,         // Default: 127.5
      threshold: 0.4,          // Default: 0.1 (confidence threshold)
      numResultsPerClass: 2,   // Default: 5 (max detections per class)
      asynch: true             // Default: true (run in background)
    );
    
    // Process results
    if (detections != null) {
      for (var detection in detections) {
        final rect = detection["rect"];
        print('${detection["detectedClass"]} - ${(detection["confidenceInClass"] * 100).toStringAsFixed(2)}%');
        print('Location: x=${rect["x"]}, y=${rect["y"]}, w=${rect["w"]}, h=${rect["h"]}');
      }
    }
  } catch (e) {
    print('Error detecting objects: $e');
  }
}
```

**Detecting Objects from Binary Data:**

```dart
Future<void> detectObjectsOnBinary(Uint8List imageBytes) async {
  try {
    List? detections = await Tflite.detectObjectOnBinary(
      binary: imageBytes,      // Required: Binary image data
      model: "SSDMobileNet",   // Default: "SSDMobileNet"
      threshold: 0.4,          // Default: 0.1
      numResultsPerClass: 2,   // Default: 5
      asynch: true             // Default: true
    );
    
    // Process results
    if (detections != null) {
      for (var detection in detections) {
        print('${detection["detectedClass"]} - ${(detection["confidenceInClass"] * 100).toStringAsFixed(2)}%');
      }
    }
  } catch (e) {
    print('Error detecting objects from binary: $e');
  }
}
```

**Detecting Objects from Camera Frames:**

> Works with [camera plugin](https://pub.dev/packages/camera). Video format: (iOS) kCVPixelFormatType_32BGRA, (Android) YUV_420_888.

```dart
Future<void> detectObjectsOnFrame(CameraImage cameraImage) async {
  try {
    List? detections = await Tflite.detectObjectOnFrame(
      bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(), // Required
      model: "SSDMobileNet",   // Default: "SSDMobileNet"
      imageHeight: cameraImage.height,
      imageWidth: cameraImage.width,
      imageMean: 127.5,        // Default: 127.5
      imageStd: 127.5,         // Default: 127.5
      rotation: 90,            // Default: 90, Android only
      numResults: 5,           // Default: 5
      threshold: 0.4,          // Default: 0.1
      asynch: true             // Default: true
    );
    
    // Process results
    if (detections != null) {
      for (var detection in detections) {
        print('${detection["detectedClass"]} - ${(detection["confidenceInClass"] * 100).toStringAsFixed(2)}%');
      }
    }
  } catch (e) {
    print('Error detecting objects on frame: $e');
  }
}
```

#### YOLO (You Only Look Once)

YOLO is another popular object detection model that's more accurate but slightly more computationally intensive than SSD MobileNet.

**Using YOLO for Object Detection:**

```dart
Future<void> detectObjectsWithYOLO(String imagePath) async {
  // YOLO-specific anchors (can be customized based on your model)
  final List<double> anchors = [
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
    5.47434, 7.88282, 3.52778, 9.77052, 9.16828
  ];
  
  try {
    List? detections = await Tflite.detectObjectOnImage(
      path: imagePath,         // Required: Path to the image file
      model: "YOLO",           // Use YOLO model
      imageMean: 0.0,          // Default: 127.5 (but YOLO typically uses 0.0)
      imageStd: 255.0,         // Default: 127.5 (but YOLO typically uses 255.0)
      threshold: 0.3,          // Default: 0.1
      numResultsPerClass: 2,   // Default: 5
      anchors: anchors,        // YOLO-specific parameter
      blockSize: 32,           // Default: 32
      numBoxesPerBlock: 5,     // Default: 5
      asynch: true             // Default: true
    );
    
    // Process results (same format as SSD MobileNet)
    if (detections != null) {
      for (var detection in detections) {
        final rect = detection["rect"];
        print('${detection["detectedClass"]} - ${(detection["confidenceInClass"] * 100).toStringAsFixed(2)}%');
        print('Location: x=${rect["x"]}, y=${rect["y"]}, w=${rect["w"]}, h=${rect["h"]}');
      }
    }
  } catch (e) {
    print('Error detecting objects with YOLO: $e');
  }
}
```

- Run on binary:

```dart
var recognitions = await Tflite.detectObjectOnBinary(
  binary: imageToByteListFloat32(resizedImage, 416, 0.0, 255.0), // required
  model: "YOLO",  
  threshold: 0.3,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
  anchors: anchors,     // defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,        // defaults to 32
  numBoxesPerBlock: 5,  // defaults to 5
  asynch: true          // defaults to true
);
```

- Run on image stream (video frame):

> Works with [camera plugin 4.0.0](https://pub.dartlang.org/packages/camera). Video format: (iOS) kCVPixelFormatType_32BGRA, (Android) YUV_420_888.

```dart
var recognitions = await Tflite.detectObjectOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  model: "YOLO",  
  imageHeight: img.height,
  imageWidth: img.width,
  imageMean: 0,         // defaults to 127.5
  imageStd: 255.0,      // defaults to 127.5
  numResults: 2,        // defaults to 5
  threshold: 0.1,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
  anchors: anchors,     // defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,        // defaults to 32
  numBoxesPerBlock: 5,  // defaults to 5
  asynch: true          // defaults to true
);
```

### Pix2Pix

> Thanks to [RP](https://github.com/shaqian/flutter_tflite/pull/18) from [Green Appers](https://github.com/GreenAppers)

- Output format:
  
  The output of Pix2Pix inference is Uint8List type. Depending on the `outputType` used, the output is:

  - (if outputType is png) byte array of a png image 

  - (otherwise) byte array of the raw output

- Run on image:

```dart
var result = await runPix2PixOnImage(
  path: filepath,       // required
  imageMean: 0.0,       // defaults to 0.0
  imageStd: 255.0,      // defaults to 255.0
  asynch: true      // defaults to true
);
```

- Run on binary:

```dart
var result = await runPix2PixOnBinary(
  binary: binary,       // required
  asynch: true      // defaults to true
);
```

- Run on image stream (video frame):

```dart
var result = await runPix2PixOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height, // defaults to 1280
  imageWidth: img.width,   // defaults to 720
  imageMean: 127.5,   // defaults to 0.0
  imageStd: 127.5,    // defaults to 255.0
  rotation: 90,       // defaults to 90, Android only
  asynch: true        // defaults to true
);
```

### Deeplab

> Thanks to [RP](https://github.com/shaqian/flutter_tflite/pull/22) from [see--](https://github.com/see--) for Android implementation.

- Output format:
  
  The output of Deeplab inference is Uint8List type. Depending on the `outputType` used, the output is:

  - (if outputType is png) byte array of a png image 

  - (otherwise) byte array of r, g, b, a values of the pixels 

- Run on image:

```dart
var result = await runSegmentationOnImage(
  path: filepath,     // required
  imageMean: 0.0,     // defaults to 0.0
  imageStd: 255.0,    // defaults to 255.0
  labelColors: [...], // defaults to https://github.com/shaqian/flutter_tflite/blob/master/lib/tflite.dart#L219
  outputType: "png",  // defaults to "png"
  asynch: true        // defaults to true
);
```

- Run on binary:

```dart
var result = await runSegmentationOnBinary(
  binary: binary,     // required
  labelColors: [...], // defaults to https://github.com/shaqian/flutter_tflite/blob/master/lib/tflite.dart#L219
  outputType: "png",  // defaults to "png"
  asynch: true        // defaults to true
);
```

- Run on image stream (video frame):

```dart
var result = await runSegmentationOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height, // defaults to 1280
  imageWidth: img.width,   // defaults to 720
  imageMean: 127.5,        // defaults to 0.0
  imageStd: 127.5,         // defaults to 255.0
  rotation: 90,            // defaults to 90, Android only
  labelColors: [...],      // defaults to https://github.com/shaqian/flutter_tflite/blob/master/lib/tflite.dart#L219
  outputType: "png",       // defaults to "png"
  asynch: true             // defaults to true
);
```

### PoseNet

> Model is from [StackOverflow thread](https://stackoverflow.com/a/55288616).

- Output format:

`x, y` are between [0, 1]. You can scale `x` by the width and `y` by the height of the image.

```
[ // array of poses/persons
  { // pose #1
    score: 0.6324902,
    keypoints: {
      0: {
        x: 0.250,
        y: 0.125,
        part: nose,
        score: 0.9971070
      },
      1: {
        x: 0.230,
        y: 0.105,
        part: leftEye,
        score: 0.9978438
      }
      ......
    }
  },
  { // pose #2
    score: 0.32534285,
    keypoints: {
      0: {
        x: 0.402,
        y: 0.538,
        part: nose,
        score: 0.8798978
      },
      1: {
        x: 0.380,
        y: 0.513,
        part: leftEye,
        score: 0.7090239
      }
      ......
    }
  },
  ......
]
```

- Run on image:

```dart
var result = await runPoseNetOnImage(
  path: filepath,     // required
  imageMean: 125.0,   // defaults to 125.0
  imageStd: 125.0,    // defaults to 125.0
  numResults: 2,      // defaults to 5
  threshold: 0.7,     // defaults to 0.5
  nmsRadius: 10,      // defaults to 20
  asynch: true        // defaults to true
);
```

- Run on binary:

```dart
var result = await runPoseNetOnBinary(
  binary: binary,     // required
  numResults: 2,      // defaults to 5
  threshold: 0.7,     // defaults to 0.5
  nmsRadius: 10,      // defaults to 20
  asynch: true        // defaults to true
);
```

- Run on image stream (video frame):

```dart
var result = await runPoseNetOnFrame(
  bytesList: img.planes.map((plane) {return plane.bytes;}).toList(),// required
  imageHeight: img.height, // defaults to 1280
  imageWidth: img.width,   // defaults to 720
  imageMean: 125.0,        // defaults to 125.0
  imageStd: 125.0,         // defaults to 125.0
  rotation: 90,            // defaults to 90, Android only
  numResults: 2,           // defaults to 5
  threshold: 0.7,          // defaults to 0.5
  nmsRadius: 10,           // defaults to 20
  asynch: true             // defaults to true
);
```

## Example

### Prediction in Static Images

  Refer to the [example](https://github.com/draz26648/tensorflow_lite_flutter/tree/master/example).

## Run test cases

`flutter test test/tflite_test.dart`