FROM alpine:latest
RUN apk add --no-cache g++ make
WORKDIR /app
COPY main.cpp Makefile ./
ENTRYPOINT ["make", "run"]
