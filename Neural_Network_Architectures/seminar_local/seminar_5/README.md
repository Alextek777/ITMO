[ClearML](https://app.clear.ml/applications)
ultralitycs

```sh
yolo train detect model=yolov8n.pt data='parking_test/data.yaml' epochs=10 
```

```sh
yolo export model=runs/detect/train6/weights/best.pt format=openvino
```

```sh
yolo predict model='runs/detect/train6/weights/best.pt' source='parking_data/images/14.png'
```


```sh
yolo predict model='runs/detect/train6/weights/best_openvino_model' source='parking_data/images/14.png'
```
