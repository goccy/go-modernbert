#include "textflag.h"

// func dot2x4Int8AVX2(x, w *int8, k, xStride, wStride int, out *int32)
//
// out[i*4+j] = Σ_{c<k} x[i*xStride+c] * w[j*wStride+c] for i in 0..1, j in
// 0..3, k a positive multiple of 32. VPMADDUBSW multiplies unsigned by
// signed bytes, so each product is formed as |x| * sign(x)·w (VPABSB /
// VPSIGNB), which keeps the pair sums within int16; VPMADDWD against ones
// widens them to int32 lanes that VPADDD accumulates.
TEXT ·dot2x4Int8AVX2(SB), NOSPLIT, $0-48
	MOVQ x+0(FP), R8
	MOVQ w+8(FP), R10
	MOVQ k+16(FP), CX
	MOVQ xStride+24(FP), R9
	MOVQ wStride+32(FP), R11
	MOVQ out+40(FP), DI
	ADDQ R8, R9          // x row 1
	LEAQ (R10)(R11*1), R12 // w row 1
	LEAQ (R12)(R11*1), R13 // w row 2
	LEAQ (R13)(R11*1), R14 // w row 3

	// Y6 = 0x0001 in every word.
	VPCMPEQW Y6, Y6, Y6
	VPSRLW   $15, Y6, Y6
	VPXOR Y8, Y8, Y8
	VPXOR Y9, Y9, Y9
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11
	VPXOR Y12, Y12, Y12
	VPXOR Y13, Y13, Y13
	VPXOR Y14, Y14, Y14
	VPXOR Y15, Y15, Y15
	SHRQ $5, CX

loop:
	VMOVDQU (R8), Y2     // x0
	VMOVDQU (R9), Y3     // x1
	VPABSB  Y2, Y0       // |x0|
	VPABSB  Y3, Y1       // |x1|

	VMOVDQU (R10), Y4    // w0
	VPSIGNB Y2, Y4, Y5
	VPMADDUBSW Y5, Y0, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y8, Y8
	VPSIGNB Y3, Y4, Y5
	VPMADDUBSW Y5, Y1, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y12, Y12

	VMOVDQU (R12), Y4    // w1
	VPSIGNB Y2, Y4, Y5
	VPMADDUBSW Y5, Y0, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y9, Y9
	VPSIGNB Y3, Y4, Y5
	VPMADDUBSW Y5, Y1, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y13, Y13

	VMOVDQU (R13), Y4    // w2
	VPSIGNB Y2, Y4, Y5
	VPMADDUBSW Y5, Y0, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y10, Y10
	VPSIGNB Y3, Y4, Y5
	VPMADDUBSW Y5, Y1, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y14, Y14

	VMOVDQU (R14), Y4    // w3
	VPSIGNB Y2, Y4, Y5
	VPMADDUBSW Y5, Y0, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y11, Y11
	VPSIGNB Y3, Y4, Y5
	VPMADDUBSW Y5, Y1, Y5
	VPMADDWD Y6, Y5, Y5
	VPADDD Y5, Y15, Y15

	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R12
	ADDQ $32, R13
	ADDQ $32, R14
	DECQ CX
	JNZ  loop

	// Horizontal sums: 8 lanes -> 1.
#define REDUCE(Y, X, off) \
	VEXTRACTI128 $1, Y, X7; \
	VPADDD X7, X, X; \
	VPHADDD X, X, X; \
	VPHADDD X, X, X; \
	VMOVD X, AX; \
	MOVL AX, off(DI)

	REDUCE(Y8, X8, 0)
	REDUCE(Y9, X9, 4)
	REDUCE(Y10, X10, 8)
	REDUCE(Y11, X11, 12)
	REDUCE(Y12, X12, 16)
	REDUCE(Y13, X13, 20)
	REDUCE(Y14, X14, 24)
	REDUCE(Y15, X15, 28)
	VZEROUPPER
	RET

// func dot4x4U8S8VNNI(xu *uint8, w *int8, k, xStride, wStride int, out *int32)
//
// out[i*4+j] = Σ_{c<k} xu[i*xStride+c] * w[j*wStride+c] for i, j in 0..3,
// k a positive multiple of 64, xu unsigned bytes (the caller offsets the
// activations by +128 and subtracts 128·Σw per weight row afterwards).
// VPDPBUSD: 4 u8×s8 products per 32-bit lane, int32 accumulation, no
// saturation. 16 zmm accumulators, folded at the end.
TEXT ·dot4x4U8S8VNNI(SB), NOSPLIT, $0-48
	MOVQ xu+0(FP), R8
	MOVQ w+8(FP), R12
	MOVQ k+16(FP), CX
	MOVQ xStride+24(FP), AX
	MOVQ wStride+32(FP), BX
	MOVQ out+40(FP), DI
	LEAQ (R8)(AX*1), R9
	LEAQ (R9)(AX*1), R10
	LEAQ (R10)(AX*1), R11
	LEAQ (R12)(BX*1), R13
	LEAQ (R13)(BX*1), R14
	LEAQ (R14)(BX*1), R15

	VPXORD Z16, Z16, Z16
	VPXORD Z17, Z17, Z17
	VPXORD Z18, Z18, Z18
	VPXORD Z19, Z19, Z19
	VPXORD Z20, Z20, Z20
	VPXORD Z21, Z21, Z21
	VPXORD Z22, Z22, Z22
	VPXORD Z23, Z23, Z23
	VPXORD Z24, Z24, Z24
	VPXORD Z25, Z25, Z25
	VPXORD Z26, Z26, Z26
	VPXORD Z27, Z27, Z27
	VPXORD Z28, Z28, Z28
	VPXORD Z29, Z29, Z29
	VPXORD Z30, Z30, Z30
	VPXORD Z31, Z31, Z31
	SHRQ $6, CX

vloop:
	VMOVDQU64 (R8), Z0
	VMOVDQU64 (R9), Z1
	VMOVDQU64 (R10), Z2
	VMOVDQU64 (R11), Z3
	VMOVDQU64 (R12), Z4
	VMOVDQU64 (R13), Z5
	VMOVDQU64 (R14), Z6
	VMOVDQU64 (R15), Z7
	VPDPBUSD Z4, Z0, Z16
	VPDPBUSD Z5, Z0, Z17
	VPDPBUSD Z6, Z0, Z18
	VPDPBUSD Z7, Z0, Z19
	VPDPBUSD Z4, Z1, Z20
	VPDPBUSD Z5, Z1, Z21
	VPDPBUSD Z6, Z1, Z22
	VPDPBUSD Z7, Z1, Z23
	VPDPBUSD Z4, Z2, Z24
	VPDPBUSD Z5, Z2, Z25
	VPDPBUSD Z6, Z2, Z26
	VPDPBUSD Z7, Z2, Z27
	VPDPBUSD Z4, Z3, Z28
	VPDPBUSD Z5, Z3, Z29
	VPDPBUSD Z6, Z3, Z30
	VPDPBUSD Z7, Z3, Z31
	ADDQ $64, R8
	ADDQ $64, R9
	ADDQ $64, R10
	ADDQ $64, R11
	ADDQ $64, R12
	ADDQ $64, R13
	ADDQ $64, R14
	ADDQ $64, R15
	DECQ CX
	JNZ  vloop

	// Fold 16 lanes: 512 -> 256 -> 128 -> horizontal adds.
// The AVX2 reductions address registers 0-15 only, so each accumulator is
// first folded into Y0.
#define VREDUCE(Z, Y, off) \
	VEXTRACTI64X4 $1, Z, Y0; \
	VPADDD Y0, Y, Y0; \
	VEXTRACTI128 $1, Y0, X1; \
	VPADDD X1, X0, X0; \
	VPHADDD X0, X0, X0; \
	VPHADDD X0, X0, X0; \
	VMOVD X0, AX; \
	MOVL AX, off(DI)

	VREDUCE(Z16, Y16, 0)
	VREDUCE(Z17, Y17, 4)
	VREDUCE(Z18, Y18, 8)
	VREDUCE(Z19, Y19, 12)
	VREDUCE(Z20, Y20, 16)
	VREDUCE(Z21, Y21, 20)
	VREDUCE(Z22, Y22, 24)
	VREDUCE(Z23, Y23, 28)
	VREDUCE(Z24, Y24, 32)
	VREDUCE(Z25, Y25, 36)
	VREDUCE(Z26, Y26, 40)
	VREDUCE(Z27, Y27, 44)
	VREDUCE(Z28, Y28, 48)
	VREDUCE(Z29, Y29, 52)
	VREDUCE(Z30, Y30, 56)
	VREDUCE(Z31, Y31, 60)
	VZEROUPPER
	RET
