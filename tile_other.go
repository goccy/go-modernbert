//go:build !arm64 && !amd64

package modernbert

import "fmt"

// selectKernels returns the kernels for k: only the Go paths exist here.
func selectKernels(k Kernel) (*kernels, error) {
	switch k {
	case KernelAuto, KernelNone:
		return &kernels{kernel: KernelNone}, nil
	}
	return nil, fmt.Errorf("modernbert: kernel %s is not available on this architecture", k)
}
