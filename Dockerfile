FROM haskell:8.10.2
RUN apt-get update -y 
RUN apt-get install libblas-dev -y
RUN apt-get install liblapack-dev -y
RUN cabal update && cabal install hlint
RUN apt-get install locales -y
RUN echo "en_US UTF-8" > /etc/locale.gen
RUN locale-gen en_GB.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_GB:en
ENV LC_ALL=en_US.UTF-8
COPY . /code
WORKDIR /code
RUN cd /code 
RUN cabal build
RUN cd ..
CMD ["bash"]
