FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
#COPY ./api /exp/api
COPY ./results /exp/results
COPY ./models /exp/models
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
#ENTRYPOINT [ "python", "question_2.py"]
CMD ["python", "app.py"]
