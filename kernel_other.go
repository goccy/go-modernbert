//go:build !arm64 && !amd64

package modernbert

func dotTile(Kernel) tile { return tile{} }
