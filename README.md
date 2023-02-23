# Luxonis Camera
Utilities to run the Luxonis object identification and publish hte results to network Tables

<H2>Docker</H2>
<H3>Build</H3>
The program is intended to run as a docker container on the target hardware (Raspberry pi).
In order to build the docker image, in a terminal (starting at the project's top level directory):
```
./build.sh
```
This would create an image TAR file (`luxonis.tar`) that you'd need to deploy over to the target hardware.

<H3>Deploy</H3>
1. Copy the tar file to the target hardware (Pi):
```
# Dont forget the ':' at the end of the command!
scp luxonis.tar pi@<Pi.IP.address>:
```

2. On the target hardware, do:
```
docker load -i luxonis.tar
```
Clean the filesystem:
```
rm luxonis.tar
```
<H3>Run</H3>
Normally, the program should run when the Pi boots up, but following an installation, you would run it manually:
```
docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw' luxonis
```
The default runtime configuration is set up to run with the Roborio: The IP addresses are hard-coded for the Roborio.
When running in test, you can change the IP addresses like:
```
docker run -e NT_IP=<ip.address> --rm --privileged -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw' luxonis 
```
