FROM ubuntu:latest

WORKDIR /root
RUN apt update -y && apt install python3-pip -y && apt install git -y
RUN git clone https://github.com/alpaka-san/stock_price_prediction.git
WORKDIR /root/stock_price_prediction
RUN mkdir $(WORKDIR)/output_dir
RUN pip install -r requirements.txt