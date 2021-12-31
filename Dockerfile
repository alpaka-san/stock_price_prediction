FROM ubuntu:latest

WORKDIR /root
RUN apt update -y && apt install python3-pip -y && apt install git -y
RUN git clone https://github.com/alpaka-san/stock_price_prediction.git
WORKDIR /root/stock_price_prediction
RUN mkdir $(WORKDIR)/output_dir
RUN pip install hydra-core
RUN pip install tensorflow
RUN pip install yahoo_finance_api2
RUN pip install hydra
RUN pip install pandas
RUN pip install matplotlib
