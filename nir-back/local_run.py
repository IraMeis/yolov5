import torch

# Model
model = torch.hub.load('D:\\nirProjectBase\\yolo\\yolov5', 'custom', 'D:\\nirProjectBase\\yolo\\yolov5\\models\\model1.pt', source='local')

# Images
im = 'C:\\Users\\Morena\\Desktop\\я2.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Results
results.print()
results.show()
