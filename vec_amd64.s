#include "textflag.h"

// func quantizeRowsU8AVX2(x *float32, k, stride int, out *byte, scales *float32, levels float32, offset int32)
//
// Eight rows of k float32 (k a multiple of 8, rows stride floats apart)
// to unsigned bytes round(x / s) + offset with s = max|x| / levels per
// row, rows written row-major k apart. Rounding is to nearest even.
TEXT ·quantizeRowsU8AVX2(SB), NOSPLIT, $0-48
	MOVQ x+0(FP), SI
	MOVQ k+8(FP), DX
	MOVQ stride+16(FP), R9
	MOVQ out+24(FP), DI
	MOVQ scales+32(FP), R10
	SHLQ $2, R9
	VBROADCASTSS levels+40(FP), Y13
	MOVL offset+44(FP), AX
	VMOVD AX, X14
	VPBROADCASTD X14, Y14
	VPCMPEQD Y15, Y15, Y15
	VPSRLD   $1, Y15, Y15 // abs mask
	MOVQ $8, R11

qrow:
	MOVQ SI, R8
	MOVQ DX, CX
	SHRQ $3, CX
	VXORPS Y0, Y0, Y0

qamax:
	VMOVUPS (R8), Y1
	VANDPS  Y15, Y1, Y1
	VMAXPS  Y1, Y0, Y0
	ADDQ    $32, R8
	DECQ    CX
	JNZ     qamax
	VEXTRACTF128 $1, Y0, X1
	VMAXPS  X1, X0, X0
	VSHUFPS $0x4E, X0, X0, X1
	VMAXPS  X1, X0, X0
	VSHUFPS $0xB1, X0, X0, X1
	VMAXPS  X1, X0, X0 // amax in lane 0
	VDIVSS  X13, X0, X2 // scale = amax / levels
	VMOVSS  X2, (R10)
	ADDQ    $4, R10
	VXORPS  X3, X3, X3
	VUCOMISS X3, X0
	JE      qzero
	VDIVSS  X0, X13, X3 // inv = levels / amax

qzero:
	VBROADCASTSS X3, Y3
	MOVQ SI, R8
	MOVQ DX, CX
	SHRQ $3, CX

qconvert:
	VMOVUPS   (R8), Y1
	VMULPS    Y3, Y1, Y1
	VCVTPS2DQ Y1, Y1
	VPADDD    Y14, Y1, Y1
	VPACKSSDW Y1, Y1, Y1
	VPERMQ    $0xD8, Y1, Y1
	VPACKUSWB Y1, Y1, Y1
	VMOVQ     X1, (DI)
	ADDQ      $32, R8
	ADDQ      $8, DI
	DECQ      CX
	JNZ       qconvert
	ADDQ R9, SI
	DECQ R11
	JNZ  qrow
	VZEROUPPER
	RET

// func quantizeProbsU8AVX2(e *float32, k, stride int, out *byte, levels float32)
//
// Eight rows of k values in [0, 1] (k a multiple of 8, rows stride floats
// apart) to bytes round(levels·e), row-major k apart.
TEXT ·quantizeProbsU8AVX2(SB), NOSPLIT, $0-36
	MOVQ e+0(FP), SI
	MOVQ k+8(FP), DX
	MOVQ stride+16(FP), R9
	MOVQ out+24(FP), DI
	SHLQ $2, R9
	VBROADCASTSS levels+32(FP), Y3
	MOVQ $8, R11

prow:
	MOVQ SI, R8
	MOVQ DX, CX
	SHRQ $3, CX

pconvert:
	VMOVUPS   (R8), Y1
	VMULPS    Y3, Y1, Y1
	VCVTPS2DQ Y1, Y1
	VPACKSSDW Y1, Y1, Y1
	VPERMQ    $0xD8, Y1, Y1
	VPACKUSWB Y1, Y1, Y1
	VMOVQ     X1, (DI)
	ADDQ      $32, R8
	ADDQ      $8, DI
	DECQ      CX
	JNZ       pconvert
	ADDQ R9, SI
	DECQ R11
	JNZ  prow
	VZEROUPPER
	RET

// func maxF32AVX2(x *float32, n int) float32
//
// The maximum of n float32 (n a positive multiple of 8).
TEXT ·maxF32AVX2(SB), NOSPLIT, $0-20
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	SHRQ $3, CX
	VMOVUPS (SI), Y0
	ADDQ $32, SI
	DECQ CX
	JZ   mdone

mloop:
	VMOVUPS (SI), Y1
	VMAXPS  Y1, Y0, Y0
	ADDQ    $32, SI
	DECQ    CX
	JNZ     mloop

mdone:
	VEXTRACTF128 $1, Y0, X1
	VMAXPS  X1, X0, X0
	VSHUFPS $0x4E, X0, X0, X1
	VMAXPS  X1, X0, X0
	VSHUFPS $0xB1, X0, X0, X1
	VMAXPS  X1, X0, X0
	VMOVSS  X0, ret+16(FP)
	VZEROUPPER
	RET

// func layerNormRowsAVX2(x *float32, T, D int, w *float32, eps float32)
//
// In place over T rows of D float32 (D a multiple of 8):
// x = (x - mean) / sqrt(var + eps) * w, var the population variance.
TEXT ·layerNormRowsAVX2(SB), NOSPLIT, $0-36
	MOVQ x+0(FP), SI
	MOVQ T+8(FP), R11
	MOVQ D+16(FP), DX
	MOVQ w+24(FP), R10
	VBROADCASTSS eps+32(FP), X10
	VCVTSI2SSQ DX, X11, X11 // D
	MOVL $0x3F800000, AX
	VMOVD AX, X12 // 1

nrow:
	MOVQ SI, R8
	MOVQ DX, CX
	SHRQ $3, CX
	VXORPS Y0, Y0, Y0

nsum:
	VMOVUPS (R8), Y1
	VADDPS  Y1, Y0, Y0
	ADDQ    $32, R8
	DECQ    CX
	JNZ     nsum
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VSHUFPS $0x4E, X0, X0, X1
	VADDPS  X1, X0, X0
	VSHUFPS $0xB1, X0, X0, X1
	VADDPS  X1, X0, X0
	VDIVSS  X11, X0, X2 // mean
	VBROADCASTSS X2, Y2
	MOVQ SI, R8
	MOVQ DX, CX
	SHRQ $3, CX
	VXORPS Y0, Y0, Y0

nvar:
	VMOVUPS (R8), Y1
	VSUBPS  Y2, Y1, Y1
	VMULPS  Y1, Y1, Y1
	VADDPS  Y1, Y0, Y0
	ADDQ    $32, R8
	DECQ    CX
	JNZ     nvar
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VSHUFPS $0x4E, X0, X0, X1
	VADDPS  X1, X0, X0
	VSHUFPS $0xB1, X0, X0, X1
	VADDPS  X1, X0, X0
	VDIVSS  X11, X0, X3
	VADDSS  X10, X3, X3
	VSQRTSS X3, X3, X3
	VDIVSS  X3, X12, X3 // 1 / sqrt(var + eps)
	VBROADCASTSS X3, Y3
	MOVQ SI, R8
	MOVQ R10, R9
	MOVQ DX, CX
	SHRQ $3, CX

nnorm:
	VMOVUPS (R8), Y1
	VSUBPS  Y2, Y1, Y1
	VMULPS  Y3, Y1, Y1
	VMULPS  (R9), Y1, Y1
	VMOVUPS Y1, (R8)
	ADDQ    $32, R8
	ADDQ    $32, R9
	DECQ    CX
	JNZ     nnorm
	MOVQ R8, SI
	DECQ R11
	JNZ  nrow
	VZEROUPPER
	RET
