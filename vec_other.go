//go:build !arm64 && !amd64

package modernbert

const hasVecNeon = false

func quantizePack8(x *float32, k, stride int, out *int8, scales *float32, levels float32) {
	panic("modernbert: quantizePack8 is arm64 only")
}

func quantizeProbs8(e *float32, k, stride int, out *uint8) {
	panic("modernbert: quantizeProbs8 is arm64 only")
}
