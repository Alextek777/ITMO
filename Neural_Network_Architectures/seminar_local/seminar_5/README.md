[clearML](https://app.clear.ml/applications) - tool to visualize learning process on dashboards

[cvat](https://app.cvat.ai/requests?page=1&pageSize=10) - tool to mark datasets

[ultralitycs git](https://github.com/ultralytics/ultralytics)
[ultralitycs web](https://www.ultralytics.com/yolo) - framework CV tools to train models using data.yaml from cli without any code

```sh
yolo train detect model=yolov8n.pt data='parking_test/data.yaml' epochs=70 
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
