from rest_framework.views import APIView

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from .serializers import RegisterSerializer
from .models import CustomUser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

from .models import MedicalImage
from .serializers import MedicalImageSerializer
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load your trained model (only once)
model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'leukemia_cnn_model.h5')
model = load_model(model_path)
class_names = ['ALL-Infected', 'Beginning', 'Healthy', 'Pre-leukemia']

class PredictAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = MedicalImageSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()

            # Load and preprocess the uploaded image
            img_path = instance.image.path
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict with CNN model
            prediction = model.predict(img_array)
            class_idx = np.argmax(prediction)
            confidence = float(np.max(prediction))
            predicted_label = class_names[class_idx]

            # Save prediction to model
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
