import 'dart:async';
import 'package:flutter/services.dart';

/// TensorFlow Lite plugin for Flutter
///
/// This class provides methods to interact with TensorFlow Lite models for
/// various machine learning tasks including image classification, object detection,
/// image-to-image translation, semantic segmentation, and pose estimation.
class Tflite {
  /// Method channel for communicating with native code
  static const MethodChannel _channel = MethodChannel('tflite');

  /// Loads a TensorFlow Lite model into memory
  ///
  /// [model] - Path to the model file (required)
  /// [labels] - Path to the labels file (optional)
  /// [numThreads] - Number of threads to use for inference (default: 1)
  /// [isAsset] - Whether the model is an asset or a file path (default: true)
  /// [useGpuDelegate] - Whether to use GPU acceleration (default: false)
  ///
  /// Returns a message indicating success or failure
  static Future<String?> loadModel({
    required String model,
    String labels = "",
    int numThreads = 1,
    bool isAsset = true,
    bool useGpuDelegate = false,
  }) async {
    return await _channel.invokeMethod(
      'loadModel',
      {
        "model": model,
        "labels": labels,
        "numThreads": numThreads,
        "isAsset": isAsset,
        'useGpuDelegate': useGpuDelegate
      },
    );
  }

  /// Runs inference on an image file for image classification
  ///
  /// [path] - Path to the image file (required)
  /// [imageMean] - Mean normalization value (default: 117.0)
  /// [imageStd] - Standard deviation normalization value (default: 1.0)
  /// [numResults] - Maximum number of results to return (default: 5)
  /// [threshold] - Minimum confidence threshold for results (default: 0.1)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of classification results, each containing:
  /// - index: The class index
  /// - label: The class label (if a labels file was provided)
  /// - confidence: The confidence score (between 0-1)
  static Future<List?> runModelOnImage({
    required String path,
    double imageMean = 117.0,
    double imageStd = 1.0,
    int numResults = 5,
    double threshold = 0.1,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runModelOnImage',
      {
        "path": path,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "numResults": numResults,
        "threshold": threshold,
        "asynch": asynch,
      },
    );
  }

  /// Runs inference on binary image data for image classification
  ///
  /// [binary] - Binary image data (required)
  /// [numResults] - Maximum number of results to return (default: 5)
  /// [threshold] - Minimum confidence threshold for results (default: 0.1)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of classification results, each containing:
  /// - index: The class index
  /// - label: The class label (if a labels file was provided)
  /// - confidence: The confidence score (between 0-1)
  static Future<List?> runModelOnBinary({
    required Uint8List binary,
    int numResults = 5,
    double threshold = 0.1,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runModelOnBinary',
      {
        "binary": binary,
        "numResults": numResults,
        "threshold": threshold,
        "asynch": asynch,
      },
    );
  }

  /// Runs inference on camera frame data for image classification
  ///
  /// [bytesList] - List of byte arrays from camera planes (required)
  /// [imageHeight] - Height of the image (default: 1280)
  /// [imageWidth] - Width of the image (default: 720)
  /// [imageMean] - Mean normalization value (default: 127.5)
  /// [imageStd] - Standard deviation normalization value (default: 127.5)
  /// [rotation] - Rotation of the image in degrees, Android only (default: 90)
  /// [numResults] - Maximum number of results to return (default: 5)
  /// [threshold] - Minimum confidence threshold for results (default: 0.1)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of classification results, each containing:
  /// - index: The class index
  /// - label: The class label (if a labels file was provided)
  /// - confidence: The confidence score (between 0-1)
  static Future<List?> runModelOnFrame({
    required List<Uint8List> bytesList,
    int imageHeight = 1280,
    int imageWidth = 720,
    double imageMean = 127.5,
    double imageStd = 127.5,
    int rotation = 90, // Android only
    int numResults = 5,
    double threshold = 0.1,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runModelOnFrame',
      {
        "bytesList": bytesList,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "rotation": rotation,
        "numResults": numResults,
        "threshold": threshold,
        "asynch": asynch,
      },
    );
  }

  /// Default anchor values for YOLO object detection model
  ///
  /// These anchors are used for the YOLO model to define the default shapes
  /// of bounding boxes at different scales
  static const List<double> anchors = [
    0.57273,
    0.677385,
    1.87446,
    2.06253,
    3.33843,
    5.47434,
    7.88282,
    3.52778,
    9.77052,
    9.16828
  ];

  /// Detects objects in an image file using either SSD MobileNet or YOLO models
  ///
  /// [path] - Path to the image file (required)
  /// [model] - Model to use: "SSDMobileNet" or "YOLO" (default: "SSDMobileNet")
  /// [imageMean] - Mean normalization value (default: 127.5)
  /// [imageStd] - Standard deviation normalization value (default: 127.5)
  /// [threshold] - Minimum confidence threshold for results (default: 0.1)
  /// [numResultsPerClass] - Maximum number of results per class (default: 5)
  /// [anchors] - Anchor values for YOLO model (default: predefined anchors)
  /// [blockSize] - Block size for YOLO model (default: 32)
  /// [numBoxesPerBlock] - Number of boxes per block for YOLO model (default: 5)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of detected objects, each containing:
  /// - detectedClass: The class label of the detected object
  /// - confidenceInClass: The confidence score (between 0-1)
  /// - rect: Object containing x, y, w, h coordinates (normalized 0-1)
  static Future<List?> detectObjectOnImage({
    required String path,
    String model = "SSDMobileNet",
    double imageMean = 127.5,
    double imageStd = 127.5,
    double threshold = 0.1,
    int numResultsPerClass = 5,
    // Used in YOLO only
    List anchors = anchors,
    int blockSize = 32,
    int numBoxesPerBlock = 5,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'detectObjectOnImage',
      {
        "path": path,
        "model": model,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "threshold": threshold,
        "numResultsPerClass": numResultsPerClass,
        "anchors": anchors,
        "blockSize": blockSize,
        "numBoxesPerBlock": numBoxesPerBlock,
        "asynch": asynch,
      },
    );
  }

  /// Detects objects in binary image data using either SSD MobileNet or YOLO models
  ///
  /// [binary] - Binary image data (required)
  /// [model] - Model to use: "SSDMobileNet" or "YOLO" (default: "SSDMobileNet")
  /// [threshold] - Minimum confidence threshold for results (default: 0.1)
  /// [numResultsPerClass] - Maximum number of results per class (default: 5)
  /// [anchors] - Anchor values for YOLO model (default: predefined anchors)
  /// [blockSize] - Block size for YOLO model (default: 32)
  /// [numBoxesPerBlock] - Number of boxes per block for YOLO model (default: 5)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of detected objects, each containing:
  /// - detectedClass: The class label of the detected object
  /// - confidenceInClass: The confidence score (between 0-1)
  /// - rect: Object containing x, y, w, h coordinates (normalized 0-1)
  static Future<List?> detectObjectOnBinary({
    required Uint8List binary,
    String model = "SSDMobileNet",
    double threshold = 0.1,
    int numResultsPerClass = 5,
    // Used in YOLO only
    List anchors = anchors,
    int blockSize = 32,
    int numBoxesPerBlock = 5,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'detectObjectOnBinary',
      {
        "binary": binary,
        "model": model,
        "threshold": threshold,
        "numResultsPerClass": numResultsPerClass,
        "anchors": anchors,
        "blockSize": blockSize,
        "numBoxesPerBlock": numBoxesPerBlock,
        "asynch": asynch,
      },
    );
  }

  /// Detects objects in camera frame data using either SSD MobileNet or YOLO models
  ///
  /// [bytesList] - List of byte arrays from camera planes (required)
  /// [model] - Model to use: "SSDMobileNet" or "YOLO" (default: "SSDMobileNet")
  /// [imageHeight] - Height of the image (default: 1280)
  /// [imageWidth] - Width of the image (default: 720)
  /// [imageMean] - Mean normalization value (default: 127.5)
  /// [imageStd] - Standard deviation normalization value (default: 127.5)
  /// [threshold] - Minimum confidence threshold for results (default: 0.1)
  /// [numResultsPerClass] - Maximum number of results per class (default: 5)
  /// [rotation] - Rotation of the image in degrees, Android only (default: 90)
  /// [anchors] - Anchor values for YOLO model (default: predefined anchors)
  /// [blockSize] - Block size for YOLO model (default: 32)
  /// [numBoxesPerBlock] - Number of boxes per block for YOLO model (default: 5)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of detected objects, each containing:
  /// - detectedClass: The class label of the detected object
  /// - confidenceInClass: The confidence score (between 0-1)
  /// - rect: Object containing x, y, w, h coordinates (normalized 0-1)
  static Future<List?> detectObjectOnFrame({
    required List<Uint8List> bytesList,
    String model = "SSDMobileNet",
    int imageHeight = 1280,
    int imageWidth = 720,
    double imageMean = 127.5,
    double imageStd = 127.5,
    double threshold = 0.1,
    int numResultsPerClass = 5,
    int rotation = 90, // Android only
    // Used in YOLO only
    List anchors = anchors,
    int blockSize = 32,
    int numBoxesPerBlock = 5,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'detectObjectOnFrame',
      {
        "bytesList": bytesList,
        "model": model,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "rotation": rotation,
        "threshold": threshold,
        "numResultsPerClass": numResultsPerClass,
        "anchors": anchors,
        "blockSize": blockSize,
        "numBoxesPerBlock": numBoxesPerBlock,
        "asynch": asynch,
      },
    );
  }

  static Future close() async {
    return await _channel.invokeMethod('close');
  }

  /// Runs Pix2Pix image-to-image translation on an image file
  ///
  /// Pix2Pix is a conditional GAN that transforms images from one domain to another
  /// (e.g., sketch to photo, day to night, etc.)
  ///
  /// [path] - Path to the image file (required)
  /// [imageMean] - Mean normalization value (default: 0)
  /// [imageStd] - Standard deviation normalization value (default: 255.0)
  /// [outputType] - Output format, either "png" or "raw" (default: "png")
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a Uint8List containing:
  /// - If outputType is "png": PNG image data that can be displayed directly
  /// - If outputType is "raw": Raw pixel data in RGBA format
  static Future<Uint8List?> runPix2PixOnImage({
    required String path,
    double imageMean = 0,
    double imageStd = 255.0,
    String outputType = "png",
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runPix2PixOnImage',
      {
        "path": path,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "outputType": outputType,
        "asynch": asynch,
      },
    );
  }

  /// Runs Pix2Pix image-to-image translation on binary image data
  ///
  /// [binary] - Binary image data (required)
  /// [outputType] - Output format, either "png" or "raw" (default: "png")
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a Uint8List containing:
  /// - If outputType is "png": PNG image data that can be displayed directly
  /// - If outputType is "raw": Raw pixel data in RGBA format
  static Future<Uint8List?> runPix2PixOnBinary({
    required Uint8List binary,
    String outputType = "png",
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runPix2PixOnBinary',
      {
        "binary": binary,
        "outputType": outputType,
        "asynch": asynch,
      },
    );
  }

  /// Runs Pix2Pix image-to-image translation on camera frame data
  ///
  /// [bytesList] - List of byte arrays from camera planes (required)
  /// [imageHeight] - Height of the image (default: 1280)
  /// [imageWidth] - Width of the image (default: 720)
  /// [imageMean] - Mean normalization value (default: 0)
  /// [imageStd] - Standard deviation normalization value (default: 255.0)
  /// [rotation] - Rotation of the image in degrees, Android only (default: 90)
  /// [outputType] - Output format, either "png" or "raw" (default: "png")
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a Uint8List containing:
  /// - If outputType is "png": PNG image data that can be displayed directly
  /// - If outputType is "raw": Raw pixel data in RGBA format
  static Future<Uint8List?> runPix2PixOnFrame({
    required List<Uint8List> bytesList,
    int imageHeight = 1280,
    int imageWidth = 720,
    double imageMean = 0,
    double imageStd = 255.0,
    int rotation = 90, // Android only
    String outputType = "png",
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runPix2PixOnFrame',
      {
        "bytesList": bytesList,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "rotation": rotation,
        "outputType": outputType,
        "asynch": asynch,
      },
    );
  }

  // https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
  /// Default color palette for Pascal VOC dataset semantic segmentation
  ///
  /// Each color represents a different class in the Pascal VOC dataset
  /// These colors are used to visualize the segmentation results
  static final List<int> pascalVOCLabelColors = [
    Color.fromARGB(255, 0, 0, 0).value, // background
    Color.fromARGB(255, 128, 0, 0).value, // aeroplane
    Color.fromARGB(255, 0, 128, 0).value, // bicycle
    Color.fromARGB(255, 128, 128, 0).value, // bird
    Color.fromARGB(255, 0, 0, 128).value, // boat
    Color.fromARGB(255, 128, 0, 128).value, // bottle
    Color.fromARGB(255, 0, 128, 128).value, // bus
    Color.fromARGB(255, 128, 128, 128).value, // car
    Color.fromARGB(255, 64, 0, 0).value, // cat
    Color.fromARGB(255, 192, 0, 0).value, // chair
    Color.fromARGB(255, 64, 128, 0).value, // cow
    Color.fromARGB(255, 192, 128, 0).value, // diningtable
    Color.fromARGB(255, 64, 0, 128).value, // dog
    Color.fromARGB(255, 192, 0, 128).value, // horse
    Color.fromARGB(255, 64, 128, 128).value, // motorbike
    Color.fromARGB(255, 192, 128, 128).value, // person
    Color.fromARGB(255, 0, 64, 0).value, // potted plant
    Color.fromARGB(255, 128, 64, 0).value, // sheep
    Color.fromARGB(255, 0, 192, 0).value, // sofa
    Color.fromARGB(255, 128, 192, 0).value, // train
    Color.fromARGB(255, 0, 64, 128).value, // tv-monitor
  ];

  /// Runs semantic segmentation on an image file
  ///
  /// Semantic segmentation classifies each pixel in an image, assigning it to a specific class
  /// (e.g., person, car, road, etc.)
  ///
  /// [path] - Path to the image file (required)
  /// [imageMean] - Mean normalization value (default: 0)
  /// [imageStd] - Standard deviation normalization value (default: 255.0)
  /// [labelColors] - List of colors to use for visualization (default: pascalVOCLabelColors)
  /// [outputType] - Output format, either "png" or "raw" (default: "png")
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a Uint8List containing:
  /// - If outputType is "png": PNG image data with colored segmentation mask
  /// - If outputType is "raw": Raw pixel data with class indices
  static Future<Uint8List?> runSegmentationOnImage({
    required String path,
    double imageMean = 0,
    double imageStd = 255.0,
    List<int>? labelColors,
    String outputType = "png",
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runSegmentationOnImage',
      {
        "path": path,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "labelColors": labelColors ?? pascalVOCLabelColors,
        "outputType": outputType,
        "asynch": asynch,
      },
    );
  }

  /// Runs semantic segmentation on binary image data
  ///
  /// [binary] - Binary image data (required)
  /// [labelColors] - List of colors to use for visualization (default: pascalVOCLabelColors)
  /// [outputType] - Output format, either "png" or "raw" (default: "png")
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a Uint8List containing:
  /// - If outputType is "png": PNG image data with colored segmentation mask
  /// - If outputType is "raw": Raw pixel data with class indices
  static Future<Uint8List?> runSegmentationOnBinary({
    required Uint8List binary,
    List<int>? labelColors,
    String outputType = "png",
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runSegmentationOnBinary',
      {
        "binary": binary,
        "labelColors": labelColors ?? pascalVOCLabelColors,
        "outputType": outputType,
        "asynch": asynch,
      },
    );
  }

  /// Runs semantic segmentation on camera frame data
  ///
  /// [bytesList] - List of byte arrays from camera planes (required)
  /// [imageHeight] - Height of the image (default: 1280)
  /// [imageWidth] - Width of the image (default: 720)
  /// [imageMean] - Mean normalization value (default: 0)
  /// [imageStd] - Standard deviation normalization value (default: 255.0)
  /// [rotation] - Rotation of the image in degrees, Android only (default: 90)
  /// [labelColors] - List of colors to use for visualization (default: pascalVOCLabelColors)
  /// [outputType] - Output format, either "png" or "raw" (default: "png")
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a Uint8List containing:
  /// - If outputType is "png": PNG image data with colored segmentation mask
  /// - If outputType is "raw": Raw pixel data with class indices
  static Future<Uint8List?> runSegmentationOnFrame({
    required List<Uint8List> bytesList,
    int imageHeight = 1280,
    int imageWidth = 720,
    double imageMean = 0,
    double imageStd = 255.0,
    int rotation = 90, // Android only
    List<int>? labelColors,
    String outputType = "png",
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runSegmentationOnFrame',
      {
        "bytesList": bytesList,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "rotation": rotation,
        "labelColors": labelColors ?? pascalVOCLabelColors,
        "outputType": outputType,
        "asynch": asynch,
      },
    );
  }

  /// Runs PoseNet human pose estimation on an image file
  ///
  /// PoseNet detects human figures in images and estimates the pose by finding
  /// body keypoints (e.g., nose, eyes, ears, shoulders, elbows, wrists, etc.)
  ///
  /// [path] - Path to the image file (required)
  /// [imageMean] - Mean normalization value (default: 127.5)
  /// [imageStd] - Standard deviation normalization value (default: 127.5)
  /// [numResults] - Maximum number of pose results to return (default: 5)
  /// [threshold] - Minimum confidence threshold for keypoints (default: 0.5)
  /// [nmsRadius] - Non-maximum suppression radius (default: 20)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of detected poses, each containing:
  /// - score: Overall confidence score for the pose
  /// - keypoints: Map of keypoint positions and confidence scores
  ///   Each keypoint contains x, y (normalized 0-1), part name, and confidence score
  static Future<List?> runPoseNetOnImage({
    required String path,
    double imageMean = 127.5,
    double imageStd = 127.5,
    int numResults = 5,
    double threshold = 0.5,
    int nmsRadius = 20,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runPoseNetOnImage',
      {
        "path": path,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "numResults": numResults,
        "threshold": threshold,
        "nmsRadius": nmsRadius,
        "asynch": asynch,
      },
    );
  }

  /// Runs PoseNet human pose estimation on binary image data
  ///
  /// [binary] - Binary image data (required)
  /// [numResults] - Maximum number of pose results to return (default: 5)
  /// [threshold] - Minimum confidence threshold for keypoints (default: 0.5)
  /// [nmsRadius] - Non-maximum suppression radius (default: 20)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of detected poses, each containing:
  /// - score: Overall confidence score for the pose
  /// - keypoints: Map of keypoint positions and confidence scores
  ///   Each keypoint contains x, y (normalized 0-1), part name, and confidence score
  static Future<List?> runPoseNetOnBinary({
    required Uint8List binary,
    int numResults = 5,
    double threshold = 0.5,
    int nmsRadius = 20,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runPoseNetOnBinary',
      {
        "binary": binary,
        "numResults": numResults,
        "threshold": threshold,
        "nmsRadius": nmsRadius,
        "asynch": asynch,
      },
    );
  }

  /// Runs PoseNet human pose estimation on camera frame data
  ///
  /// [bytesList] - List of byte arrays from camera planes (required)
  /// [imageHeight] - Height of the image (default: 1280)
  /// [imageWidth] - Width of the image (default: 720)
  /// [imageMean] - Mean normalization value (default: 127.5)
  /// [imageStd] - Standard deviation normalization value (default: 127.5)
  /// [rotation] - Rotation of the image in degrees, Android only (default: 90)
  /// [numResults] - Maximum number of pose results to return (default: 5)
  /// [threshold] - Minimum confidence threshold for keypoints (default: 0.5)
  /// [nmsRadius] - Non-maximum suppression radius (default: 20)
  /// [asynch] - Whether to run inference asynchronously (default: true)
  ///
  /// Returns a list of detected poses, each containing:
  /// - score: Overall confidence score for the pose
  /// - keypoints: Map of keypoint positions and confidence scores
  ///   Each keypoint contains x, y (normalized 0-1), part name, and confidence score
  static Future<List?> runPoseNetOnFrame({
    required List<Uint8List> bytesList,
    int imageHeight = 1280,
    int imageWidth = 720,
    double imageMean = 127.5,
    double imageStd = 127.5,
    int rotation = 90, // Android only
    int numResults = 5,
    double threshold = 0.5,
    int nmsRadius = 20,
    bool asynch = true,
  }) async {
    return await _channel.invokeMethod(
      'runPoseNetOnFrame',
      {
        "bytesList": bytesList,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "rotation": rotation,
        "numResults": numResults,
        "threshold": threshold,
        "nmsRadius": nmsRadius,
        "asynch": asynch,
      },
    );
  }
}
