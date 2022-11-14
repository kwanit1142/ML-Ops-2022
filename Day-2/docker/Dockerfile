FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./api /exp/api
COPY ./best_svm_model.joblib /exp/best_svm_model.joblib
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
