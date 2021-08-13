FROM ubuntu 

COPY ./input-data/* /usr/share/ehrudite/input/
RUN apt-get update && apt-get -y install python python3-pip git

RUN git clone https://github.com/ClaudioBorges/ehrpreper /tmp/ehrpreper
WORKDIR /tmp/ehrpreper
RUN python3 setup.py install

WORKDIR /tmp/ehrudite
RUN ehrpreper -i /usr/share/ehrudite/input/mimicIII/ -o /tmp/ehrudite -x mimicIII -vvv
