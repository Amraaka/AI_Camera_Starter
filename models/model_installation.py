from ultralytics import YOLO

# Load the YOLOv11-x PyTorch model
model = YOLO("yolo26m.pt")  # This will download the model automatically

# Export the model to ONNX format
# 'dynamic=True' allows variable input image sizes
# 'simplify=True' optimizes the graph
model.export(format="onnx", dynamic=True, simplify=True)

# The output file will be named 'yolo11x.onnx'
    