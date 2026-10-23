#include "textflag.h"

// func dot4x4Int8(x, w *int8, k, xStride, wStride int, out *int32)
//
// out[i*4+j] = Σ_{c<k} x[i*xStride+c] * w[j*wStride+c] for i, j in 0..3,
// k a positive multiple of 16, via SDOT (dot product of four int8 pairs per
// 32-bit lane; dotprod feature). 16 accumulators, one per output, each
// holding four partial sums that ADDV folds at the end.
TEXT ·dot4x4Int8(SB), NOSPLIT, $0-48
	MOVD x+0(FP), R0
	MOVD w+8(FP), R4
	MOVD k+16(FP), R8
	MOVD xStride+24(FP), R9
	MOVD wStride+32(FP), R10
	MOVD out+40(FP), R11

	// x rows: R0..R3; w rows: R4..R7.
	ADD  R9, R0, R1
	ADD  R9, R1, R2
	ADD  R9, R2, R3
	ADD  R10, R4, R5
	ADD  R10, R5, R6
	ADD  R10, R6, R7

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	VEOR V24.B16, V24.B16, V24.B16
	VEOR V25.B16, V25.B16, V25.B16
	VEOR V26.B16, V26.B16, V26.B16
	VEOR V27.B16, V27.B16, V27.B16
	VEOR V28.B16, V28.B16, V28.B16
	VEOR V29.B16, V29.B16, V29.B16
	VEOR V30.B16, V30.B16, V30.B16
	VEOR V31.B16, V31.B16, V31.B16

	LSR  $4, R8, R8 // k / 16 steps

loop:
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V1.B16]
	VLD1.P 16(R2), [V2.B16]
	VLD1.P 16(R3), [V3.B16]
	VLD1.P 16(R4), [V4.B16]
	VLD1.P 16(R5), [V5.B16]
	VLD1.P 16(R6), [V6.B16]
	VLD1.P 16(R7), [V7.B16]

	// sdot vD.4s, vN.16b, vM.16b = 0x4E809400 | M<<16 | N<<5 | D
	// row 0 (V0) × w rows V4..V7 -> V16..V19
	WORD $0x4E849410 // sdot v16.4s, v0.16b, v4.16b
	WORD $0x4E859411 // sdot v17.4s, v0.16b, v5.16b
	WORD $0x4E869412 // sdot v18.4s, v0.16b, v6.16b
	WORD $0x4E879413 // sdot v19.4s, v0.16b, v7.16b
	// row 1 (V1) -> V20..V23
	WORD $0x4E849434 // sdot v20.4s, v1.16b, v4.16b
	WORD $0x4E859435 // sdot v21.4s, v1.16b, v5.16b
	WORD $0x4E869436 // sdot v22.4s, v1.16b, v6.16b
	WORD $0x4E879437 // sdot v23.4s, v1.16b, v7.16b
	// row 2 (V2) -> V24..V27
	WORD $0x4E849458 // sdot v24.4s, v2.16b, v4.16b
	WORD $0x4E859459 // sdot v25.4s, v2.16b, v5.16b
	WORD $0x4E86945A // sdot v26.4s, v2.16b, v6.16b
	WORD $0x4E87945B // sdot v27.4s, v2.16b, v7.16b
	// row 3 (V3) -> V28..V31
	WORD $0x4E84947C // sdot v28.4s, v3.16b, v4.16b
	WORD $0x4E85947D // sdot v29.4s, v3.16b, v5.16b
	WORD $0x4E86947E // sdot v30.4s, v3.16b, v6.16b
	WORD $0x4E87947F // sdot v31.4s, v3.16b, v7.16b

	SUBS $1, R8, R8
	BNE  loop

	// Fold each accumulator's four lanes and store.
	VADDV V16.S4, V16
	VADDV V17.S4, V17
	VADDV V18.S4, V18
	VADDV V19.S4, V19
	VADDV V20.S4, V20
	VADDV V21.S4, V21
	VADDV V22.S4, V22
	VADDV V23.S4, V23
	VADDV V24.S4, V24
	VADDV V25.S4, V25
	VADDV V26.S4, V26
	VADDV V27.S4, V27
	VADDV V28.S4, V28
	VADDV V29.S4, V29
	VADDV V30.S4, V30
	VADDV V31.S4, V31
	VMOV V16.S[0], R12
	MOVW R12, 0(R11)
	VMOV V17.S[0], R12
	MOVW R12, 4(R11)
	VMOV V18.S[0], R12
	MOVW R12, 8(R11)
	VMOV V19.S[0], R12
	MOVW R12, 12(R11)
	VMOV V20.S[0], R12
	MOVW R12, 16(R11)
	VMOV V21.S[0], R12
	MOVW R12, 20(R11)
	VMOV V22.S[0], R12
	MOVW R12, 24(R11)
	VMOV V23.S[0], R12
	MOVW R12, 28(R11)
	VMOV V24.S[0], R12
	MOVW R12, 32(R11)
	VMOV V25.S[0], R12
	MOVW R12, 36(R11)
	VMOV V26.S[0], R12
	MOVW R12, 40(R11)
	VMOV V27.S[0], R12
	MOVW R12, 44(R11)
	VMOV V28.S[0], R12
	MOVW R12, 48(R11)
	VMOV V29.S[0], R12
	MOVW R12, 52(R11)
	VMOV V30.S[0], R12
	MOVW R12, 56(R11)
	VMOV V31.S[0], R12
	MOVW R12, 60(R11)
	RET
