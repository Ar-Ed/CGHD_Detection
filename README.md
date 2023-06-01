## To do
- [x] Dataset Conversion
  - [x] PASCAL VOC -> YOLO Format
  - [x] PASCAL VOC -> COCO Format
- [] Colab Compatibility
- [x] Detectron2
  - [x] Faster RCNN R50, R101
  - [x] Cascade RCNN R50
  - [x] Retinanet R50, R101
- [x] DETR
  - [x] DETR R50, R50-DC5
- [x] YOLO
  - [x] YOLOv8s, YOLOv8l
  - [x] YOLOv5s6, YOLOv5l6  

## Some facts
- All Augmentation is turned off
- Best performing model on val set is saved (Choose: AP50, AP, Loss)
- Parameters are set for RTX3060 6G 


## Expected Directory Structure
- setup_environment.ipynb
- train_detectron.ipynb
- train_detr.ipynb
- train_yolo.ipynb
- utils_detectron2.py
- utils_environment.py
- README.md

- CGHD-1152.yaml
- CGHD-1152-YOLO/
- CGHD-1152/
- CGHD-COCO/
- content/

- detectron_models/
- yolov5_models/
- yolov8_models/
  
- detectron2/
- detr/
- yolov5/
- voc2coco/
  
- yolov5l6.pt
- yolov5s.pt
- yolov8l.pt
- yolov8n.pt
- yolov8s.pt
