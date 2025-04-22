import os
import numpy as np
import gdown
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from .models import MedicalImage, CustomUser
from .serializers import MedicalImageSerializer, RegisterSerializer

# Prevent TensorFlow from using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = None  # Global model reference

def ensure_model_loaded():
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'leukemia_cnn_model.h5')

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(
            'https://drive.google.com/uc?id=1Ak-FBeJrJPaecD15hMScwZpAM7JEVSMN',
            model_path,
            quiet=False
        )

    if model is None:
        model = load_model(model_path)

class_names = ['ALL-Infected', 'Beginning', 'Healthy', 'Pre-leukemia']

class PredictAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        ensure_model_loaded()

        serializer = MedicalImageSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()

            img_path = instance.image.path
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            class_idx = np.argmax(prediction)
            confidence = float(np.max(prediction))
            predicted_label = class_names[class_idx]

            instance.prediction = f"{predicted_label} ({confidence:.2%})"
            instance.save()

            return Response(MedicalImageSerializer(instance).data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, _ = Token.objects.get_or_create(user=user)

            response_data = {
                "token": token.key,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "phone_number": user.phone_number,
                }
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")
        user = authenticate(request, email=email, password=password)

        if user:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                "token": token.key,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "full_name": user.full_name,
                    "phone_number": user.phone_number
                }
            })
        return Response({"error": "Invalid credentials"}, status=status.HTTP_400_BAD_REQUEST)
