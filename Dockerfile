FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Pre-download the BLIP model at build time so cold starts are fast
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
    BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
    BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')"

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
