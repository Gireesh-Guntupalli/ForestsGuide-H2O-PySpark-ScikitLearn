FROM python:3.8.10
RUN apt-get update && apt-get install build-essential swig python-dev -y && \
    pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir seaborn
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir yellowbrick
