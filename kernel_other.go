//go:build !arm64 && !amd64

package modernbert

func tileKernel() tile { return tile{} }
