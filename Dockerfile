FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
