//go:build arm64

package modernbert

import (
	"testing"

	"golang.org/x/sys/cpu"
)

// TestKernelAutoWithoutI8MM checks that KernelAuto falls back to the Go
// paths on an arm64 CPU without the i8mm extension instead of installing
// the SMMLA tile.
func TestKernelAutoWithoutI8MM(t *testing.T) {
	saved := cpu.ARM64.HasI8MM
	cpu.ARM64.HasI8MM = false
	defer func() { cpu.ARM64.HasI8MM = saved }()
	ks, err := selectKernels(KernelAuto)
	if err != nil {
		t.Fatal(err)
	}
	if ks.kernel != KernelNone || ks.tile != nil {
		t.Fatalf("KernelAuto without i8mm resolved to %s with tile %v", ks.kernel, ks.tile != nil)
	}
}
