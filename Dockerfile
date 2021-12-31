FROM ubuntu:latest

RUN apt update -y && apt install python3-pip -y
RUN pip install hydra-core
RUN pip install tensorflow
RUN pip install yahoo_finance_api2
RUN pip install hydra
RUN pip install pandas
RUN pip install matplotlib
