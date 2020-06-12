# BBoxerwGradCAM

This class forms boundary boxes (rectangle and polygon) using GradCAM outputs for a given image.

The purpose of this class is to develop Rectangle and Polygon coordinates that define an object based on an image classification model. 

The 'automatic' creation of these coordinates, which are often included in COCO JSONs used to train object detection models, is valuable because data preparation and labeling can be a time consuming task.

This class takes 5 user inputs:
* Pretrained Learner (image classification model)
* GradCAM Heatmap (heatmap of GradCAM object - formed by a pretrained image classification learner)
* Source Image
* Image Resizing Scale (also applied to corresponding GradCAM heatmap)
* BBOX Rectangle Resizing Scale
* Class is compatible with google colab and other Python 3 enivronments
