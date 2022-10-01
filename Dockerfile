FROM python:3.10
WORKDIR /rotation_correction_service
COPY ./service-requirements.txt /rotation_correction_service/requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --upgrade -r /rotation_correction_service/requirements.txt
COPY ./models/rotation_net.onnx /rotation_correction_service/mdoels/rotation_net.onnx
COPY ./src/prediction.py /rotation_correction_service/src/prediction.py
COPY ./service.py /rotation_correction_service/service.py

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "9000"]