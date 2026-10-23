#include "textflag.h"

// Constant table: each entry is one float32 repeated four times.

DATA mathConsts<>+0(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+4(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+8(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+12(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+16(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+20(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+24(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+28(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+32(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+36(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+40(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+44(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+48(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+52(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+56(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+60(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+64(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+68(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+72(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+76(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+80(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+84(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+88(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+92(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+96(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+100(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+104(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+108(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+112(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+116(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+120(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+124(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+128(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+132(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+136(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+140(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+144(SB)/4, $0x3F800000 // one
DATA mathConsts<>+148(SB)/4, $0x3F800000 // one
DATA mathConsts<>+152(SB)/4, $0x3F800000 // one
DATA mathConsts<>+156(SB)/4, $0x3F800000 // one
DATA mathConsts<>+160(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+164(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+168(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+172(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+176(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+180(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+184(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+188(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+192(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+196(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+200(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+204(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+208(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+212(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+216(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+220(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+224(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+228(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+232(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+236(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+240(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+244(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+248(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+252(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+256(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+260(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+264(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+268(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+272(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+276(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+280(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+284(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+288(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+292(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+296(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+300(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+304(SB)/4, $0x3F000000 // half
DATA mathConsts<>+308(SB)/4, $0x3F000000 // half
DATA mathConsts<>+312(SB)/4, $0x3F000000 // half
DATA mathConsts<>+316(SB)/4, $0x3F000000 // half
GLOBL mathConsts<>(SB), RODATA, $320

// func softmaxExpNeon(x *float32, n int, max float32) float32
//
// x[i] = exp(x[i] - max) in place for n values (n a positive multiple of
// 4); returns their sum. exp as in math32.go: clamp, range reduction by
// ln2 with the exponent rebuilt in the float bits, a degree-7 polynomial.
TEXT ·softmaxExpNeon(SB), NOSPLIT, $0-28
	MOVD  x+0(FP), R0
	MOVD  n+8(FP), R1
	FMOVS max+16(FP), F8
	LSR   $2, R1, R1
	MOVD $mathConsts<>(SB), R10
	VLD1.P 64(R10), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R10), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R10), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R10), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1.P 64(R10), [V28.S4, V29.S4, V30.S4, V31.S4]
	WORD $0x4E040508 // dup v8.4s, v8.s[0]
	VEOR V9.B16, V9.B16, V9.B16 // sum

sloop:
	VLD1 (R0), [V0.S4]
	WORD $0x4EA8D400 // fsub v0.4s, v0.4s, v8.4s
	WORD $0x4E36C401 // fmaxnm v1.4s, v0.4s, v22.4s
	WORD $0x4EB7C421 // fminnm v1.4s, v1.4s, v23.4s
	WORD $0x6E2CDC22 // fmul v2.4s, v1.4s, v12.4s
	WORD $0x4E218842 // frintn v2.4s, v2.4s
	WORD $0x4EADCC41 // fmls v1.4s, v2.4s, v13.4s
	WORD $0x4EAECC41 // fmls v1.4s, v2.4s, v14.4s
	VMOV V20.B16, V3.B16
	WORD $0x6E21DC63 // fmul v3.4s, v3.4s, v1.4s
	WORD $0x4E33D463 // fadd v3.4s, v3.4s, v19.4s
	WORD $0x6E21DC63 // fmul v3.4s, v3.4s, v1.4s
	WORD $0x4E32D463 // fadd v3.4s, v3.4s, v18.4s
	WORD $0x6E21DC63 // fmul v3.4s, v3.4s, v1.4s
	WORD $0x4E31D463 // fadd v3.4s, v3.4s, v17.4s
	WORD $0x6E21DC63 // fmul v3.4s, v3.4s, v1.4s
	WORD $0x4E30D463 // fadd v3.4s, v3.4s, v16.4s
	WORD $0x6E21DC63 // fmul v3.4s, v3.4s, v1.4s
	WORD $0x4E2FD463 // fadd v3.4s, v3.4s, v15.4s
	WORD $0x6E21DC63 // fmul v3.4s, v3.4s, v1.4s
	WORD $0x4E35D463 // fadd v3.4s, v3.4s, v21.4s
	WORD $0x6E21DC63 // fmul v3.4s, v3.4s, v1.4s
	WORD $0x4E35D463 // fadd v3.4s, v3.4s, v21.4s
	WORD $0x4EA1B842 // fcvtzs v2.4s, v2.4s
	VSHL $23, V2.S4, V2.S4
	VADD V2.S4, V3.S4, V1.S4
	VST1.P [V1.S4], 16(R0)
	WORD $0x4E21D529 // fadd v9.4s, v9.4s, v1.4s
	SUBS $1, R1, R1
	BNE  sloop
	WORD $0x6E29D529 // faddp v9.4s, v9.4s, v9.4s
	WORD $0x6E29D529 // faddp v9.4s, v9.4s, v9.4s
	FMOVS F9, ret+24(FP)
	RET

// func gegluErfNeon(a, x, gate *float32, n int)
//
// a[i] = gelu_erf(x[i]) * g[i] for n values (n a positive multiple of 4),
// gelu_erf(x) = x/2 · (1 + erf(x/√2)) with erf by A&S 7.1.26.
TEXT ·gegluErfNeon(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD x+8(FP), R1
	MOVD gate+16(FP), R2
	MOVD n+24(FP), R3
	LSR  $2, R3, R3
	MOVD $mathConsts<>(SB), R10
	VLD1.P 64(R10), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R10), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R10), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R10), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1.P 64(R10), [V28.S4, V29.S4, V30.S4, V31.S4]
	MOVW $0x80000000, R4
	VDUP R4, V11.S4 // sign mask

gloop:
	VLD1.P 16(R1), [V0.S4] // x
	WORD $0x4EA0F801 // fabs v1.4s, v0.4s
	WORD $0x6E3EDC21 // fmul v1.4s, v1.4s, v30.4s
	VMOV V21.B16, V2.B16
	WORD $0x4E38CC22 // fmla v2.4s, v1.4s, v24.4s
	WORD $0x6E22FEA2 // fdiv v2.4s, v21.4s, v2.4s
	VMOV V29.B16, V3.B16
	WORD $0x6E22DC63 // fmul v3.4s, v3.4s, v2.4s
	WORD $0x4E3CD463 // fadd v3.4s, v3.4s, v28.4s
	WORD $0x6E22DC63 // fmul v3.4s, v3.4s, v2.4s
	WORD $0x4E3BD463 // fadd v3.4s, v3.4s, v27.4s
	WORD $0x6E22DC63 // fmul v3.4s, v3.4s, v2.4s
	WORD $0x4E3AD463 // fadd v3.4s, v3.4s, v26.4s
	WORD $0x6E22DC63 // fmul v3.4s, v3.4s, v2.4s
	WORD $0x4E39D463 // fadd v3.4s, v3.4s, v25.4s
	WORD $0x6E22DC63 // fmul v3.4s, v3.4s, v2.4s
	WORD $0x6E21DC24 // fmul v4.4s, v1.4s, v1.4s
	WORD $0x6EA0F884 // fneg v4.4s, v4.4s
	WORD $0x4E36C485 // fmaxnm v5.4s, v4.4s, v22.4s
	WORD $0x4EB7C4A5 // fminnm v5.4s, v5.4s, v23.4s
	WORD $0x6E2CDCA6 // fmul v6.4s, v5.4s, v12.4s
	WORD $0x4E2188C6 // frintn v6.4s, v6.4s
	WORD $0x4EADCCC5 // fmls v5.4s, v6.4s, v13.4s
	WORD $0x4EAECCC5 // fmls v5.4s, v6.4s, v14.4s
	VMOV V20.B16, V7.B16
	WORD $0x6E25DCE7 // fmul v7.4s, v7.4s, v5.4s
	WORD $0x4E33D4E7 // fadd v7.4s, v7.4s, v19.4s
	WORD $0x6E25DCE7 // fmul v7.4s, v7.4s, v5.4s
	WORD $0x4E32D4E7 // fadd v7.4s, v7.4s, v18.4s
	WORD $0x6E25DCE7 // fmul v7.4s, v7.4s, v5.4s
	WORD $0x4E31D4E7 // fadd v7.4s, v7.4s, v17.4s
	WORD $0x6E25DCE7 // fmul v7.4s, v7.4s, v5.4s
	WORD $0x4E30D4E7 // fadd v7.4s, v7.4s, v16.4s
	WORD $0x6E25DCE7 // fmul v7.4s, v7.4s, v5.4s
	WORD $0x4E2FD4E7 // fadd v7.4s, v7.4s, v15.4s
	WORD $0x6E25DCE7 // fmul v7.4s, v7.4s, v5.4s
	WORD $0x4E35D4E7 // fadd v7.4s, v7.4s, v21.4s
	WORD $0x6E25DCE7 // fmul v7.4s, v7.4s, v5.4s
	WORD $0x4E35D4E7 // fadd v7.4s, v7.4s, v21.4s
	WORD $0x4EA1B8C6 // fcvtzs v6.4s, v6.4s
	VSHL $23, V6.S4, V6.S4
	VADD V6.S4, V7.S4, V5.S4
	VMOV V21.B16, V6.B16
	WORD $0x4EA5CC66 // fmls v6.4s, v3.4s, v5.4s
	VAND V11.B16, V0.B16, V7.B16
	VORR V7.B16, V6.B16, V6.B16
	WORD $0x4E35D4C6 // fadd v6.4s, v6.4s, v21.4s
	WORD $0x6E20DCC6 // fmul v6.4s, v6.4s, v0.4s
	WORD $0x6E3FDCC6 // fmul v6.4s, v6.4s, v31.4s
	VLD1.P 16(R2), [V8.S4]
	WORD $0x6E28DCC6 // fmul v6.4s, v6.4s, v8.4s
	VST1.P [V6.S4], 16(R0)
	SUBS   $1, R3, R3
	BNE    gloop
	RET
