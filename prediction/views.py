from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from .serializers import RegisterSerializer, MedicalImageSerializer
from .models import CustomUser, MedicalImage

import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------------
# 🔽 Google Drive Model Download
# -----------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use

GOOGLE_DRIVE_FILE_ID = '1Ak-FBeJrJPaecD15hMScwZpAM7JEVSMN'
MODEL_FILENAME = 'leukemia_cnn_model.h5'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'ml_model')
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("✅ Model downloaded successfully!")

download_model_from_drive()
model = load_model(MODEL_PATH)
class_names = ['ALL-Infected', 'Beginning', 'Healthy', 'Pre-leukemia']

# -----------------------------------
# 🔍 Predict API
# -----------------------------------
class PredictAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
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

# -----------------------------------
# 👤 Register API
# -----------------------------------
class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, _ = Token.objects.get_or_create(user=user)
            return Response({
                "token": token.key,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "phone_number": user.phone_number,
                }
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# -----------------------------------
# 🔐 Login API
# -----------------------------------
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
