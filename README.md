# ballot-detector
This is ballot image detector and counter, for image processing class in our master degree. Hopefully used by our gov.

## preparation
require pull submodule
```
git submodule update --init
```

## how to run
- server
```
python project.py
```

- client (raspberry pi)
need to change ip in this file to server ip
```
python raspberry_pi_camera_client.py
```
