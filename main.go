package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"math"
	"time"

	"gocv.io/x/gocv"
	"gocv.io/x/gocv/cuda"
)

var (
	modelFile      = flag.String("model", "yolov5s.onnx", "model path")
	modelImageSize = flag.Int("size", 640, "model image size")
	srcImage       = flag.String("image", "images/face.jpg", "input image")
)

func printDevices() {
	num := cuda.GetCudaEnabledDeviceCount()
	for i := 0; i < num; i++ {
		cuda.PrintCudaDeviceInfo(i)
	}
}

func main() {
	flag.Parse()

	printDevices()

	net := gocv.ReadNetFromONNX(*modelFile)
	net.SetPreferableBackend(gocv.NetBackendCUDA)
	net.SetPreferableTarget(gocv.NetTargetCUDA)

	src := gocv.IMRead(*srcImage, gocv.IMReadColor)
	modelSize := image.Pt(*modelImageSize, *modelImageSize)

	resized := gocv.NewMat()

	letterBox(src, &resized, modelSize)

	blob := gocv.BlobFromImage(resized, 1/255.0, modelSize, gocv.Scalar{}, true, false)

	unconnectedLayerIds := net.GetUnconnectedOutLayers()
	layerNames := []string{}
	for _, id := range unconnectedLayerIds {
		layer := net.GetLayer(id)
		layerNames = append(layerNames, layer.GetName())
	}

	var outs []gocv.Mat
	for i := 0; i < 100; i++ {
		// detect 100 times
		start := time.Now()
		net.SetInput(blob, "")
		outs = net.ForwardLayers(layerNames)
		end := time.Now()
		fmt.Println(i+1, "cost", end.Sub(start))
	}

	sz := outs[0].Size()
	rows := sz[1]
	cols := sz[2]

	ptr, _ := outs[0].DataPtrFloat32()

	boxes := []image.Rectangle{}
	scores := []float32{}
	indices := []int{}
	classIndexLists := []int{}

	for j := 0; j < rows; j++ {
		i0 := j * cols
		i1 := j*cols + cols
		line := ptr[i0:i1]
		x := line[0]
		y := line[1]
		w := line[2]
		h := line[3]
		sc := line[4]
		confs := line[5:]
		bestId, bestScore := getBestFromConfs(confs)
		bestScore *= sc

		scores = append(scores, bestScore)
		boxes = append(boxes, image.Rect(int(x-w/2), int(y-w/2), int(x+w/2), int(y+h/2)))
		indices = append(indices, -1)
		classIndexLists = append(classIndexLists, bestId)
	}

	fmt.Println("Do NMS in", len(boxes), "boxes")
	gocv.NMSBoxes(boxes, scores, 0.25, 0.45, indices)

	nmsNumber := 0
	goodBoxes := []image.Rectangle{}
	goodScores := []float32{}
	goodClassIds := []int{}

	output := resized.Clone()

	for _, v := range indices {
		if v < 0 {
			break
		} else {
			nmsNumber++
			goodBoxes = append(goodBoxes, boxes[v])
			goodScores = append(goodScores, scores[v])
			goodClassIds = append(goodClassIds, classIndexLists[v])

			gocv.Rectangle(&output, boxes[v], color.RGBA{0, 255, 0, 255}, 3)
			gocv.PutText(&output, fmt.Sprintf("ClassId: %d, Score: %f", classIndexLists[v], scores[v]),
				boxes[v].Min, gocv.FontHersheySimplex, 1, color.RGBA{255, 0, 0, 255}, 2)
		}
	}

	fmt.Println("After NMS", nmsNumber, "keeped")

	w := gocv.NewWindow("detected")
	w.ResizeWindow(modelSize.X, modelSize.Y)
	w.IMShow(output)
	w.WaitKey(-1)

	src.Close()
}

func getBestFromConfs(confs []float32) (int, float32) {
	bestId := 0
	bestScore := float32(0)
	for i, v := range confs {
		if v > bestScore {
			bestId = i
			bestScore = v
		}
	}
	return bestId, bestScore
}

func letterBox(src gocv.Mat, dst *gocv.Mat, size image.Point) {
	k := math.Min(float64(size.X)/float64(src.Cols()), float64(size.Y)/float64(src.Rows()))
	newSize := image.Pt(int(k*float64(src.Cols())), int(k*float64(src.Rows())))

	tmp := gocv.NewMat()
	gocv.Resize(src, &tmp, newSize, 0, 0, gocv.InterpolationLinear)

	if dst.Cols() != size.X || dst.Rows() != size.Y {
		dstNew := gocv.NewMatWithSize(size.Y, size.X, src.Type())
		dstNew.CopyTo(dst)
	}

	rectOfTmp := image.Rect(0, 0, newSize.X, newSize.Y)

	regionOfDst := dst.Region(rectOfTmp)
	tmp.CopyTo(&regionOfDst)
}
