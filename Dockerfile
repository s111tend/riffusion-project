FROM python:3.9
WORKDIR /app

RUN apt-get update && apt-get install

RUN pip install numpy
RUN pip install typing_extensions

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/hmartiro/riffusion-inference
RUN pip install --upgrade pillow
RUN pip install --upgrade gradio

COPY app.py .

CMD ["python", "app.py"]