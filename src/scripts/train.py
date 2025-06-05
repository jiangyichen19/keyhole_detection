from ultralytics import YOLO


# Load a model
# model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("/home/sophgo/Code/yichen/keyhole_detection/src/yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("/home/sophgo/Code/yichen/keyhole_detection/src/ultralytics/ultralytics/cfg/models/11/yolo11x.yaml")
results = model.train(data="/home/sophgo/Code/yichen/keyhole_detection/src/ultralytics/ultralytics/cfg/datasets/key_hole_dataset.yaml",
                      batch = 1,imgsz=640,device = 0,epochs = 200,project = "../src/result",
                      resume = True,
                      name="yolo11x_epoch{epochs}_batch{batch}_size{imgsz}_model-{modelsize}".format(
                        epochs=200,
                        batch=1,
                        imgsz=640,
                        modelsize = 'x'
                    ))
# 使用GPU训练