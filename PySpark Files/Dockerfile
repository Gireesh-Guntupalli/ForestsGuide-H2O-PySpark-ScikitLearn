ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.8.10

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

RUN apt-get update && apt-get install build-essential swig python-dev -y && \
    pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir seaborn
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir scikit-learn
ARG PYSPARK_VERSION=3.2.0
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}

