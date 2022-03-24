FROM tensorflow/tensorflow:2.8.0-gpu

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH="${PATH}:/root/.poetry/bin"

COPY . /opt/ae-sentence-embeddings
WORKDIR /opt/ae-sentence-embeddings

RUN poetry build && \
    poetry install

RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["./docker/entrypoint.sh"]
