#include "textflag.h"

// func gemmSmmla8(a *int8, b *int8, k, ncb, bStride int, xs, ws *float32, out *float32, outStride int)
//
// One block of eight activation rows (a, packed by packInt8Blocks8) times
// ncb blocks of eight weight rows (b, the same packing, bStride bytes
// apart), k a multiple of 8: out[r*outStride+c] = xs[r] * ws[c] * Σ
// a[r]·b[c] as float32. Sixteen SMMLA accumulators carry the 8×8 int32 tile
// over the whole k; the scales apply once per output.
TEXT ·gemmSmmla8(SB), NOSPLIT, $0-72
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD k+16(FP), R2
	MOVD ncb+24(FP), R3
	MOVD bStride+32(FP), R12
	MOVD xs+40(FP), R4
	MOVD ws+48(FP), R5
	MOVD out+56(FP), R6
	MOVD outStride+64(FP), R7
	LSL  $2, R7, R7 // row stride in bytes
	LSR  $3, R2, R2 // k / 8 steps
	VLD1 (R4), [V12.S4, V13.S4] // xs: rows 0..3, 4..7

colblock:
	MOVD R0, R8
	MOVD R1, R9
	MOVD R2, R10
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

kloop:
	VLD1.P 64(R8), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 64(R9), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4E84A410 // smmla v16.4s, v0.16b, v4.16b
	WORD $0x4E85A411 // smmla v17.4s, v0.16b, v5.16b
	WORD $0x4E86A412 // smmla v18.4s, v0.16b, v6.16b
	WORD $0x4E87A413 // smmla v19.4s, v0.16b, v7.16b
	WORD $0x4E84A434 // smmla v20.4s, v1.16b, v4.16b
	WORD $0x4E85A435 // smmla v21.4s, v1.16b, v5.16b
	WORD $0x4E86A436 // smmla v22.4s, v1.16b, v6.16b
	WORD $0x4E87A437 // smmla v23.4s, v1.16b, v7.16b
	WORD $0x4E84A458 // smmla v24.4s, v2.16b, v4.16b
	WORD $0x4E85A459 // smmla v25.4s, v2.16b, v5.16b
	WORD $0x4E86A45A // smmla v26.4s, v2.16b, v6.16b
	WORD $0x4E87A45B // smmla v27.4s, v2.16b, v7.16b
	WORD $0x4E84A47C // smmla v28.4s, v3.16b, v4.16b
	WORD $0x4E85A47D // smmla v29.4s, v3.16b, v5.16b
	WORD $0x4E86A47E // smmla v30.4s, v3.16b, v6.16b
	WORD $0x4E87A47F // smmla v31.4s, v3.16b, v7.16b
	SUBS $1, R10, R10
	BNE  kloop

	// Epilogue: rows 2i (zip1) and 2i+1 (zip2) of the 2×2 tiles, to
	// float32, times the column scales and the row scale.
	VLD1.P 32(R5), [V14.S4, V15.S4]
	MOVD   R6, R11
	WORD $0x4ED13A08 // zip1 v8.2d, v16.2d, v17.2d
	WORD $0x4ED33A49 // zip1 v9.2d, v18.2d, v19.2d
	WORD $0x4ED17A0A // zip2 v10.2d, v16.2d, v17.2d
	WORD $0x4ED37A4B // zip2 v11.2d, v18.2d, v19.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8C9108 // fmul v8.4s, v8.4s, v12.s[0]
	WORD $0x4F8C9129 // fmul v9.4s, v9.4s, v12.s[0]
	WORD $0x4FAC914A // fmul v10.4s, v10.4s, v12.s[1]
	WORD $0x4FAC916B // fmul v11.4s, v11.4s, v12.s[1]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	WORD $0x4ED53A88 // zip1 v8.2d, v20.2d, v21.2d
	WORD $0x4ED73AC9 // zip1 v9.2d, v22.2d, v23.2d
	WORD $0x4ED57A8A // zip2 v10.2d, v20.2d, v21.2d
	WORD $0x4ED77ACB // zip2 v11.2d, v22.2d, v23.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8C9908 // fmul v8.4s, v8.4s, v12.s[2]
	WORD $0x4F8C9929 // fmul v9.4s, v9.4s, v12.s[2]
	WORD $0x4FAC994A // fmul v10.4s, v10.4s, v12.s[3]
	WORD $0x4FAC996B // fmul v11.4s, v11.4s, v12.s[3]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	WORD $0x4ED93B08 // zip1 v8.2d, v24.2d, v25.2d
	WORD $0x4EDB3B49 // zip1 v9.2d, v26.2d, v27.2d
	WORD $0x4ED97B0A // zip2 v10.2d, v24.2d, v25.2d
	WORD $0x4EDB7B4B // zip2 v11.2d, v26.2d, v27.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8D9108 // fmul v8.4s, v8.4s, v13.s[0]
	WORD $0x4F8D9129 // fmul v9.4s, v9.4s, v13.s[0]
	WORD $0x4FAD914A // fmul v10.4s, v10.4s, v13.s[1]
	WORD $0x4FAD916B // fmul v11.4s, v11.4s, v13.s[1]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	WORD $0x4EDD3B88 // zip1 v8.2d, v28.2d, v29.2d
	WORD $0x4EDF3BC9 // zip1 v9.2d, v30.2d, v31.2d
	WORD $0x4EDD7B8A // zip2 v10.2d, v28.2d, v29.2d
	WORD $0x4EDF7BCB // zip2 v11.2d, v30.2d, v31.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8D9908 // fmul v8.4s, v8.4s, v13.s[2]
	WORD $0x4F8D9929 // fmul v9.4s, v9.4s, v13.s[2]
	WORD $0x4FAD994A // fmul v10.4s, v10.4s, v13.s[3]
	WORD $0x4FAD996B // fmul v11.4s, v11.4s, v13.s[3]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	ADD  $32, R6, R6
	ADD  R12, R1, R1
	SUBS $1, R3, R3
	BNE  colblock
	RET

// func gemmUsmmla8(a *uint8, b *int8, k, ncb, bStride int, xs, ws *float32, out *float32, outStride int)
//
// One block of eight activation rows (a, packed by packInt8Blocks8) times
// ncb blocks of eight weight rows (b, the same packing, bStride bytes
// apart), k a multiple of 8: out[r*outStride+c] = xs[r] * ws[c] * Σ
// a[r]·b[c] as float32. Sixteen USMMLA accumulators carry the 8×8 int32 tile
// over the whole k; the scales apply once per output.
TEXT ·gemmUsmmla8(SB), NOSPLIT, $0-72
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD k+16(FP), R2
	MOVD ncb+24(FP), R3
	MOVD bStride+32(FP), R12
	MOVD xs+40(FP), R4
	MOVD ws+48(FP), R5
	MOVD out+56(FP), R6
	MOVD outStride+64(FP), R7
	LSL  $2, R7, R7 // row stride in bytes
	LSR  $3, R2, R2 // k / 8 steps
	VLD1 (R4), [V12.S4, V13.S4] // xs: rows 0..3, 4..7

colblock:
	MOVD R0, R8
	MOVD R1, R9
	MOVD R2, R10
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

kloop:
	VLD1.P 64(R8), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P 64(R9), [V4.B16, V5.B16, V6.B16, V7.B16]
	WORD $0x4E84AC10 // usmmla v16.4s, v0.16b, v4.16b
	WORD $0x4E85AC11 // usmmla v17.4s, v0.16b, v5.16b
	WORD $0x4E86AC12 // usmmla v18.4s, v0.16b, v6.16b
	WORD $0x4E87AC13 // usmmla v19.4s, v0.16b, v7.16b
	WORD $0x4E84AC34 // usmmla v20.4s, v1.16b, v4.16b
	WORD $0x4E85AC35 // usmmla v21.4s, v1.16b, v5.16b
	WORD $0x4E86AC36 // usmmla v22.4s, v1.16b, v6.16b
	WORD $0x4E87AC37 // usmmla v23.4s, v1.16b, v7.16b
	WORD $0x4E84AC58 // usmmla v24.4s, v2.16b, v4.16b
	WORD $0x4E85AC59 // usmmla v25.4s, v2.16b, v5.16b
	WORD $0x4E86AC5A // usmmla v26.4s, v2.16b, v6.16b
	WORD $0x4E87AC5B // usmmla v27.4s, v2.16b, v7.16b
	WORD $0x4E84AC7C // usmmla v28.4s, v3.16b, v4.16b
	WORD $0x4E85AC7D // usmmla v29.4s, v3.16b, v5.16b
	WORD $0x4E86AC7E // usmmla v30.4s, v3.16b, v6.16b
	WORD $0x4E87AC7F // usmmla v31.4s, v3.16b, v7.16b
	SUBS $1, R10, R10
	BNE  kloop

	// Epilogue: rows 2i (zip1) and 2i+1 (zip2) of the 2×2 tiles, to
	// float32, times the column scales and the row scale.
	VLD1.P 32(R5), [V14.S4, V15.S4]
	MOVD   R6, R11
	WORD $0x4ED13A08 // zip1 v8.2d, v16.2d, v17.2d
	WORD $0x4ED33A49 // zip1 v9.2d, v18.2d, v19.2d
	WORD $0x4ED17A0A // zip2 v10.2d, v16.2d, v17.2d
	WORD $0x4ED37A4B // zip2 v11.2d, v18.2d, v19.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8C9108 // fmul v8.4s, v8.4s, v12.s[0]
	WORD $0x4F8C9129 // fmul v9.4s, v9.4s, v12.s[0]
	WORD $0x4FAC914A // fmul v10.4s, v10.4s, v12.s[1]
	WORD $0x4FAC916B // fmul v11.4s, v11.4s, v12.s[1]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	WORD $0x4ED53A88 // zip1 v8.2d, v20.2d, v21.2d
	WORD $0x4ED73AC9 // zip1 v9.2d, v22.2d, v23.2d
	WORD $0x4ED57A8A // zip2 v10.2d, v20.2d, v21.2d
	WORD $0x4ED77ACB // zip2 v11.2d, v22.2d, v23.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8C9908 // fmul v8.4s, v8.4s, v12.s[2]
	WORD $0x4F8C9929 // fmul v9.4s, v9.4s, v12.s[2]
	WORD $0x4FAC994A // fmul v10.4s, v10.4s, v12.s[3]
	WORD $0x4FAC996B // fmul v11.4s, v11.4s, v12.s[3]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	WORD $0x4ED93B08 // zip1 v8.2d, v24.2d, v25.2d
	WORD $0x4EDB3B49 // zip1 v9.2d, v26.2d, v27.2d
	WORD $0x4ED97B0A // zip2 v10.2d, v24.2d, v25.2d
	WORD $0x4EDB7B4B // zip2 v11.2d, v26.2d, v27.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8D9108 // fmul v8.4s, v8.4s, v13.s[0]
	WORD $0x4F8D9129 // fmul v9.4s, v9.4s, v13.s[0]
	WORD $0x4FAD914A // fmul v10.4s, v10.4s, v13.s[1]
	WORD $0x4FAD916B // fmul v11.4s, v11.4s, v13.s[1]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	WORD $0x4EDD3B88 // zip1 v8.2d, v28.2d, v29.2d
	WORD $0x4EDF3BC9 // zip1 v9.2d, v30.2d, v31.2d
	WORD $0x4EDD7B8A // zip2 v10.2d, v28.2d, v29.2d
	WORD $0x4EDF7BCB // zip2 v11.2d, v30.2d, v31.2d
	WORD $0x4E21D908 // scvtf v8.4s, v8.4s
	WORD $0x4E21D929 // scvtf v9.4s, v9.4s
	WORD $0x4E21D94A // scvtf v10.4s, v10.4s
	WORD $0x4E21D96B // scvtf v11.4s, v11.4s
	WORD $0x6E2EDD08 // fmul v8.4s, v8.4s, v14.4s
	WORD $0x6E2FDD29 // fmul v9.4s, v9.4s, v15.4s
	WORD $0x6E2EDD4A // fmul v10.4s, v10.4s, v14.4s
	WORD $0x6E2FDD6B // fmul v11.4s, v11.4s, v15.4s
	WORD $0x4F8D9908 // fmul v8.4s, v8.4s, v13.s[2]
	WORD $0x4F8D9929 // fmul v9.4s, v9.4s, v13.s[2]
	WORD $0x4FAD994A // fmul v10.4s, v10.4s, v13.s[3]
	WORD $0x4FAD996B // fmul v11.4s, v11.4s, v13.s[3]
	VST1 [V8.S4, V9.S4], (R11)
	ADD  R7, R11, R11
	VST1 [V10.S4, V11.S4], (R11)
	ADD  R7, R11, R11
	ADD  $32, R6, R6
	ADD  R12, R1, R1
	SUBS $1, R3, R3
	BNE  colblock
	RET

