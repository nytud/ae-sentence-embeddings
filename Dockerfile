FROM tensorflow/tensorflow:2.9.1-gpu

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH="${PATH}:/root/.poetry/bin"

ARG USER_NAME
ARG GROUP_NAME
ARG UID
ARG GID
RUN groupadd -g ${GID} ${GROUP_NAME}
RUN useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${GID} -G sudo -u ${UID} ${USER_NAME}

COPY . /home/${USER_NAME}/ae-sentence-embeddings
WORKDIR /home/${USER_NAME}/ae-sentence-embeddings

RUN poetry config virtualenvs.in-project true && \
    poetry build && \
    poetry install

RUN chown -R ${UID}:${GID} /home/${USER_NAME}
RUN chmod 775 docker/entrypoint.sh
ENTRYPOINT ["./docker/entrypoint.sh"]
