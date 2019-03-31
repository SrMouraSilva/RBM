#########################
# Tensorflow base
#########################
#FROM tensorflow/tensorflow:1.13.1-py3
#
#WORKDIR /app
#
#RUN apt-get install python3-venv --yes --no-install-recommends

#########################
# Alpine base
#########################
#python:3.6-alpine
#
# Dependencies
#RUN apk add --no-cache bash
#
#CMD ["/bin/bash"]


#########################
# Ubuntu base
#########################
FROM ubuntu:18.10

RUN apt-get update \
 && apt-get install python3-venv --yes --no-install-recommends

WORKDIR /app
