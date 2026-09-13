from ultralytics import YOLO

if __name__ == "__main__":

    for model_name in ["yolov8" + name + "-seg.pt" for name in ["n", "s", "m", "l", "x"]]:

        data = "data.yaml"
        imgsz = 512
        epochs = 20
        batch = 8
        device = "0"

        model = YOLO(model_name)

        model_name = model_name.replace(".pt", "")

        results = model.train(
            data=data,
            epochs=epochs,
            batch=batch,
            device=device,
            imgsz=imgsz,
            save=True,
            project="head_thorax_segmentation_with_YOLOv8\\train",
            name=model_name
        )

        model = YOLO(f'head_thorax_segmentation_with_YOLOv8\\train\\{model_name}\\weights\\best.pt')
        metrics = model.val(
            data=data,
            imgsz=imgsz,
            batch=batch,
            device=device,
            save_json=True,
            split="test",
            project="head_thorax_segmentation_with_YOLOv8\\val",
            name=model_name
        )
        print("--------------------------------")
        print(metrics)
