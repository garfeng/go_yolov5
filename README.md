# Use YoloV5 with Golang. CUDA avaiable on Windows

`YoloV5` `GoCV` `Golang` `CUDA` `Windows`

## Dependences

* OpenCV build with CUDA
* GoCV (cpp source build via Visual Studio. You could get it here: https://github.com/garfeng/gocv/blob/cuda_win/UseCudaOnWindows.md)

## Step

1. Download the code
``` shell
git clone https://github.com/garfeng/go_yolov5.git
```

2. Build

Before build, please ensure that, there are 3 dlls in `C:/opencv/cvglue/bin`

``` shell
admin@pc MINGW64 /c/opencv/cvglue
$ tree
.
├── bin
│   ├── gocv.dll 
│   ├── gocv_contrib.dll
│   └── gocv_cuda.dll
└── lib
    ├── gocv.lib
    ├── gocv_contrib.lib
    └── gocv_cuda.lib

2 directories, 6 files
```

Then build go_yolov5.

``` shell
cd go_yolov5
go build
cp /c/opencv/cvglue/bin/gocv.dll ./
cp /c/opencv/cvglue/bin/gocv_cuda.dll ./
```

3. Run it
``` shell
./go_yolov5.exe
```

It equals:
```
./go_yolov5.exe -model yolov5s.onnx -size 640 -image images/face.jpg
```

Or run go_yolov5.exe with your own model and images.

``` shell
./go_yolov5.exe -model <model path> -size <size> -image <input image>
```

