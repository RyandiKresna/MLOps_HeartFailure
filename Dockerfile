FROM tensorflow/serving:latest

COPY ./serving_model /models
ENV MODEL_NAME=Heart-Failure-Prediction