#/bin/sh

docker build -t luxonis .

docker save luxonis -o luxonis.tar
