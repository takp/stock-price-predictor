FROM ubuntu:18.04

RUN apt-get update \
&& apt-get install -y python3.6 \
&& apt-get install -y python3-pip \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* \
&& cd /usr/local/bin \
&& ln -s /usr/bin/python3 python \
&& pip3 install --upgrade pip

COPY ./requirements.txt /root/requirements.txt
RUN pip3 install -r /root/requirements.txt

# work directory
COPY . /var/app/
WORKDIR /var/app/

CMD python main.py
