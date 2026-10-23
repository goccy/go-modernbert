#include "textflag.h"

// func quantizePack8(x *float32, k, stride int, out *int8, scales *float32, levels float32)
//
// Eight rows of k float32 (k a multiple of 8, rows stride floats apart)
// to symmetric int8, one scale per row (max|x|/levels, rounding to
// nearest, ties away), written in the eight-row block layout of
// packInt8Blocks8: row r's values 8s..8s+7 at out[64s+8r].
TEXT ·quantizePack8(SB), NOSPLIT, $0-44
	MOVD  x+0(FP), R0
	MOVD  k+8(FP), R1
	MOVD  stride+16(FP), R10
	MOVD  out+24(FP), R2
	MOVD  scales+32(FP), R3
	FMOVS levels+40(FP), F3
	LSR   $3, R1, R1 // k / 8 steps
	LSL   $2, R10, R10 // row stride in bytes
	VEOR  V4.B16, V4.B16, V4.B16 // 0
	MOVD  $0, R5

row:
	MOVD R1, R9
	MOVD R0, R6
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16

amax:
	VLD1.P 32(R6), [V0.S4, V1.S4]
	WORD $0x4EA0F800 // fabs v0, v0
	WORD $0x4EA0F821 // fabs v1, v1
	WORD $0x4E20C4A5 // fmaxnm v5.4s, v5.4s, v0.4s
	WORD $0x4E21C4C6 // fmaxnm v6.4s, v6.4s, v1.4s
	SUBS $1, R9, R9
	BNE  amax
	WORD $0x4E26C4A5 // fmaxnm v5.4s, v5.4s, v6.4s
	WORD $0x6E30C8A2 // fmaxnmv s, v2, v5
	FDIVS  F3, F2, F7 // scale = amax / levels
	FDIVS  F2, F3, F5 // inv = levels / amax
	FCMPS  F4, F2
	FCSELS EQ, F4, F5, F5 // inv = 0 when amax == 0
	FMOVS  F7, (R3)
	ADD    $4, R3, R3
	WORD $0x4E0404A7 // dup .4s, s[0] v7, v5
	MOVD R1, R9
	MOVD R0, R6
	MOVD R2, R7

convert:
	VLD1.P 32(R6), [V0.S4, V1.S4]
	WORD $0x6E27DC00 // fmul v0.4s, v0.4s, v7.4s
	WORD $0x6E27DC21 // fmul v1.4s, v1.4s, v7.4s
	WORD $0x4E21C800 // fcvtas v0, v0
	WORD $0x4E21C821 // fcvtas v1, v1
	WORD $0x0E614800 // sqxtn v0.4h, v0.4s
	WORD $0x4E614820 // sqxtn2 v0.8h, v1.4s
	WORD $0x0E214800 // sqxtn v0.8b, v0.8h
	FMOVD F0, (R7)
	ADD   $64, R7, R7
	SUBS  $1, R9, R9
	BNE   convert
	ADD   R10, R0, R0
	ADD   $8, R2, R2
	ADD   $1, R5, R5
	CMP   $8, R5
	BNE   row
	RET

// func quantizeProbs8(e *float32, k, stride int, out *uint8)
//
// Eight rows of k values in [0, 1] (k a multiple of 8, rows stride floats
// apart) to unsigned 8-bit as round(255·e), in the eight-row block layout
// of packInt8Blocks8.
TEXT ·quantizeProbs8(SB), NOSPLIT, $0-32
	MOVD  e+0(FP), R0
	MOVD  k+8(FP), R1
	MOVD  stride+16(FP), R10
	MOVD  out+24(FP), R2
	LSR   $3, R1, R1
	LSL   $2, R10, R10
	MOVW  $0x437F0000, R8
	FMOVS R8, F7 // 255
	WORD $0x4E0404E7 // dup .4s, s[0] v7, v7
	MOVD $0, R5

prow:
	MOVD R1, R9
	MOVD R0, R6
	MOVD R2, R7

pconvert:
	VLD1.P 32(R6), [V0.S4, V1.S4]
	WORD $0x6E27DC00 // fmul v0.4s, v0.4s, v7.4s
	WORD $0x6E27DC21 // fmul v1.4s, v1.4s, v7.4s
	WORD $0x6E21C800 // fcvtau v0, v0
	WORD $0x6E21C821 // fcvtau v1, v1
	WORD $0x2E614800 // uqxtn v0.4h, v0.4s
	WORD $0x6E614820 // uqxtn2 v0.8h, v1.4s
	WORD $0x2E214800 // uqxtn v0.8b, v0.8h
	FMOVD F0, (R7)
	ADD   $64, R7, R7
	SUBS  $1, R9, R9
	BNE   pconvert
	ADD   R10, R0, R0
	ADD   $8, R2, R2
	ADD   $1, R5, R5
	CMP   $8, R5
	BNE   prow
	RET

// func maxF32x4(x *float32, n int) float32
//
// The maximum of n float32 (n a positive multiple of 4).
TEXT ·maxF32x4(SB), NOSPLIT, $0-20
	MOVD x+0(FP), R0
	MOVD n+8(FP), R1
	LSR  $2, R1, R1
	VLD1.P 16(R0), [V0.S4]
	SUBS $1, R1, R1
	BEQ  maxdone

maxloop:
	VLD1.P 16(R0), [V1.S4]
	WORD $0x4E21C400 // fmaxnm v0.4s, v0.4s, v1.4s
	SUBS $1, R1, R1
	BNE  maxloop

maxdone:
	WORD $0x6E30C800 // fmaxnmv s, v0, v0
	FMOVS F0, ret+16(FP)
	RET

// func layerNormRowsNeon(x *float32, T, D int, w *float32, eps float32)
//
// In place over T rows of D float32 (D a multiple of 8):
// x = (x - mean) / sqrt(var + eps) * w, var the population variance.
TEXT ·layerNormRowsNeon(SB), NOSPLIT, $0-36
	MOVD   x+0(FP), R0
	MOVD   T+8(FP), R1
	MOVD   D+16(FP), R2
	MOVD   w+24(FP), R3
	FMOVS  eps+32(FP), F10
	LSR    $3, R2, R4 // D / 8 steps
	SCVTFS R2, F11    // D
	MOVW   $0x3F800000, R8
	FMOVS  R8, F14 // 1

normrow:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	MOVD R0, R5
	MOVD R4, R6

sum:
	VLD1.P 32(R5), [V2.S4, V3.S4]
	WORD $0x4E22D400 // fadd v0.4s, v0.4s, v2.4s
	WORD $0x4E23D421 // fadd v1.4s, v1.4s, v3.4s
	SUBS $1, R6, R6
	BNE  sum
	WORD $0x4E21D400 // fadd v0.4s, v0.4s, v1.4s
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	FDIVS F11, F0, F12 // mean
	WORD $0x4E04058C // dup .4s, s[0] v12, v12
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	MOVD R0, R5
	MOVD R4, R6

variance:
	VLD1.P 32(R5), [V2.S4, V3.S4]
	WORD $0x4EACD442 // fsub v2.4s, v2.4s, v12.4s
	WORD $0x4EACD463 // fsub v3.4s, v3.4s, v12.4s
	WORD $0x4E22CC40 // fmla v0.4s, v2.4s, v2.4s
	WORD $0x4E23CC61 // fmla v1.4s, v3.4s, v3.4s
	SUBS $1, R6, R6
	BNE  variance
	WORD $0x4E21D400 // fadd v0.4s, v0.4s, v1.4s
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	FDIVS  F11, F0, F13
	FADDS  F10, F13, F13
	FSQRTS F13, F13
	FDIVS  F13, F14, F13 // 1 / sqrt(var + eps)
	WORD $0x4E0405AD // dup .4s, s[0] v13, v13
	MOVD R0, R5
	MOVD R3, R7
	MOVD R4, R6

normalize:
	VLD1   (R5), [V2.S4, V3.S4]
	VLD1.P 32(R7), [V4.S4, V5.S4]
	WORD $0x4EACD442 // fsub v2.4s, v2.4s, v12.4s
	WORD $0x4EACD463 // fsub v3.4s, v3.4s, v12.4s
	WORD $0x6E2DDC42 // fmul v2.4s, v2.4s, v13.4s
	WORD $0x6E2DDC63 // fmul v3.4s, v3.4s, v13.4s
	WORD $0x6E24DC42 // fmul v2.4s, v2.4s, v4.4s
	WORD $0x6E25DC63 // fmul v3.4s, v3.4s, v5.4s
	VST1.P [V2.S4, V3.S4], 32(R5)
	SUBS   $1, R6, R6
	BNE    normalize
	MOVD   R5, R0
	SUBS   $1, R1, R1
	BNE    normrow
	RET
