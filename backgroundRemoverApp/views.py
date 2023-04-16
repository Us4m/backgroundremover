from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import plot_one_box

class ImageProcessing(APIView):
    def post(self, request, format=None):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            # Load YOLOv8 model
            model = attempt_load('yolov8.pt', map_location=torch.device('cpu'))

            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Read image from request
            image = request.FILES['image']
            img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            # Resize image to model input size
            img = cv2.resize(img, (640, 640))

            # Convert image to tensor
            img = torch.from_numpy(img).to(device).float()
            img /= 255.0
            img = img.unsqueeze(0)

            # Make predictions
            pred = model(img)[0]

            # Apply non-maximum suppression
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

            # Get bounding box coordinates and class labels
            boxes = []
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = det[:, :4].clone().cpu().detach().numpy()
                    boxes.extend(det[:, :4])

            # Create mask
            mask = np.zeros_like(img.cpu().squeeze().numpy(), dtype=np.uint8)
            for box in boxes:
                box = box.astype(np.int32)
                mask = cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)

            # Apply mask to image
            result = cv2.bitwise_and(img.cpu().squeeze().numpy(), mask)

            # Read new background image
            background_image = cv2.imread('background.jpg')

            # Resize background image to match the size of the result image
            background_image = cv2.resize(background_image, (result.shape[1], result.shape[0]))

            # Apply new background to the result image
            result = cv2.addWeighted(result, 1, background_image, 1, 0)

            # Save result as PNG file
            cv2.imwrite('result.png', result * 255.0)

            # Return the result
