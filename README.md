# [Original Repo](https://github.com/PingoLH/FCHarDNet.git)

The Sematic Segmentation Model Predicts 19 Classes of Cityscape Dataset. This code works on cpu as well.

### Requirements

`pip install -r requirement.txt`

### Pretrained Weigths

* [Cityscape](https://github.com/dlision/Sematic_Segmentaion/blob/main/weights/hardnet70_cityscapes_model.pkl)


### Sample Command to Run Inference on Folder of Images

`python test.py --model_path weights/hardnet70_cityscapes_model.pkl --input sample_input/ --output sample_output/`

### Prediction Sample

![alt text](https://github.com/dlision/Sematic_Segmentaion/blob/main/sample_output/sample.jpg)
