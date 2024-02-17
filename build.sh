#/bin/sh

docker build --platform linux/arm64 -t frc4048-luxonis .

docker save frc4048-luxonis -o frc4048-luxonis.tar
