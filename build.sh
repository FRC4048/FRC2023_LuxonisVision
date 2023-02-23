#/bin/sh

docker build --platform linux/arm64 -t luxonis .

docker save luxonis -o luxonis.tar
