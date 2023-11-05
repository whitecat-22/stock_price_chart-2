FROM public.ecr.aws/lambda/python:3.11
ENV AWS_DEFAULT_REGION ap-northeast-1

# install build libs
RUN yum groupinstall -y "Development Tools" \
    && yum install -y which openssl

RUN python3 -m pip install numpy

COPY ./requirements.txt /opt/
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /opt/requirements.txt -t /var/task

WORKDIR /var/task
COPY handler.py .
# COPY slack.py .
# COPY twitter.py .

CMD [ "handler.lambdahandler" ]

