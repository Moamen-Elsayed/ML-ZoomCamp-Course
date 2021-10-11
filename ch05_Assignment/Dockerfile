FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["test.py", "./"]


EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:9696", "test:app" ]