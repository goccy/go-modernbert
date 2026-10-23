#include "textflag.h"

// Constant table: each entry is one float32 repeated eight times.
DATA mathConsts<>+0(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+4(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+8(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+12(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+16(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+20(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+24(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+28(SB)/4, $0x3FB8AA3B // log2e
DATA mathConsts<>+32(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+36(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+40(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+44(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+48(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+52(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+56(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+60(SB)/4, $0x3F317200 // ln2hi
DATA mathConsts<>+64(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+68(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+72(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+76(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+80(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+84(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+88(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+92(SB)/4, $0x35BFBE8E // ln2lo
DATA mathConsts<>+96(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+100(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+104(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+108(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+112(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+116(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+120(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+124(SB)/4, $0x3F000000 // c2
DATA mathConsts<>+128(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+132(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+136(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+140(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+144(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+148(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+152(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+156(SB)/4, $0x3E2AAAAB // c3
DATA mathConsts<>+160(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+164(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+168(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+172(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+176(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+180(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+184(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+188(SB)/4, $0x3D2AAAAB // c4
DATA mathConsts<>+192(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+196(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+200(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+204(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+208(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+212(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+216(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+220(SB)/4, $0x3C088889 // c5
DATA mathConsts<>+224(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+228(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+232(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+236(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+240(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+244(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+248(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+252(SB)/4, $0x3AB60B61 // c6
DATA mathConsts<>+256(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+260(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+264(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+268(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+272(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+276(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+280(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+284(SB)/4, $0x39500D01 // c7
DATA mathConsts<>+288(SB)/4, $0x3F800000 // one
DATA mathConsts<>+292(SB)/4, $0x3F800000 // one
DATA mathConsts<>+296(SB)/4, $0x3F800000 // one
DATA mathConsts<>+300(SB)/4, $0x3F800000 // one
DATA mathConsts<>+304(SB)/4, $0x3F800000 // one
DATA mathConsts<>+308(SB)/4, $0x3F800000 // one
DATA mathConsts<>+312(SB)/4, $0x3F800000 // one
DATA mathConsts<>+316(SB)/4, $0x3F800000 // one
DATA mathConsts<>+320(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+324(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+328(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+332(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+336(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+340(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+344(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+348(SB)/4, $0xC2AE0000 // lo
DATA mathConsts<>+352(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+356(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+360(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+364(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+368(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+372(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+376(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+380(SB)/4, $0x42B00000 // hi
DATA mathConsts<>+384(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+388(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+392(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+396(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+400(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+404(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+408(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+412(SB)/4, $0x3EA7BA05 // p
DATA mathConsts<>+416(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+420(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+424(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+428(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+432(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+436(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+440(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+444(SB)/4, $0x3E827906 // a1
DATA mathConsts<>+448(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+452(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+456(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+460(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+464(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+468(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+472(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+476(SB)/4, $0xBE91A98E // a2
DATA mathConsts<>+480(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+484(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+488(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+492(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+496(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+500(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+504(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+508(SB)/4, $0x3FB5F0E3 // a3
DATA mathConsts<>+512(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+516(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+520(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+524(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+528(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+532(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+536(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+540(SB)/4, $0xBFBA00E3 // a4
DATA mathConsts<>+544(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+548(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+552(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+556(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+560(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+564(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+568(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+572(SB)/4, $0x3F87DC22 // a5
DATA mathConsts<>+576(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+580(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+584(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+588(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+592(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+596(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+600(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+604(SB)/4, $0x3F3504F3 // rsqrt2
DATA mathConsts<>+608(SB)/4, $0x3F000000 // half
DATA mathConsts<>+612(SB)/4, $0x3F000000 // half
DATA mathConsts<>+616(SB)/4, $0x3F000000 // half
DATA mathConsts<>+620(SB)/4, $0x3F000000 // half
DATA mathConsts<>+624(SB)/4, $0x3F000000 // half
DATA mathConsts<>+628(SB)/4, $0x3F000000 // half
DATA mathConsts<>+632(SB)/4, $0x3F000000 // half
DATA mathConsts<>+636(SB)/4, $0x3F000000 // half
DATA mathConsts<>+640(SB)/4, $0x80000000 // signmask
DATA mathConsts<>+644(SB)/4, $0x80000000 // signmask
DATA mathConsts<>+648(SB)/4, $0x80000000 // signmask
DATA mathConsts<>+652(SB)/4, $0x80000000 // signmask
DATA mathConsts<>+656(SB)/4, $0x80000000 // signmask
DATA mathConsts<>+660(SB)/4, $0x80000000 // signmask
DATA mathConsts<>+664(SB)/4, $0x80000000 // signmask
DATA mathConsts<>+668(SB)/4, $0x80000000 // signmask
GLOBL mathConsts<>(SB), RODATA, $672

// func softmaxExpAVX2(x *float32, n int, max float32) float32
//
// x[i] = exp(x[i] - max) in place for n values (n a positive multiple of
// 8); returns their sum. exp as in math32.go.
TEXT ·softmaxExpAVX2(SB), NOSPLIT, $0-28
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	VBROADCASTSS max+16(FP), Y8
	SHRQ $3, CX
	VXORPS Y9, Y9, Y9

sloop:
	VMOVUPS (SI), Y0
	VSUBPS  Y8, Y0, Y0
	VMAXPS mathConsts<>+320(SB), Y0, Y1
	VMINPS mathConsts<>+352(SB), Y1, Y1
	VMULPS mathConsts<>+0(SB), Y1, Y2
	VROUNDPS $0, Y2, Y2
	VMULPS mathConsts<>+32(SB), Y2, Y3
	VSUBPS Y3, Y1, Y1
	VMULPS mathConsts<>+64(SB), Y2, Y3
	VSUBPS Y3, Y1, Y1
	VMOVUPS mathConsts<>+256(SB), Y3
	VMULPS Y1, Y3, Y3
	VADDPS mathConsts<>+224(SB), Y3, Y3
	VMULPS Y1, Y3, Y3
	VADDPS mathConsts<>+192(SB), Y3, Y3
	VMULPS Y1, Y3, Y3
	VADDPS mathConsts<>+160(SB), Y3, Y3
	VMULPS Y1, Y3, Y3
	VADDPS mathConsts<>+128(SB), Y3, Y3
	VMULPS Y1, Y3, Y3
	VADDPS mathConsts<>+96(SB), Y3, Y3
	VMULPS Y1, Y3, Y3
	VADDPS mathConsts<>+288(SB), Y3, Y3
	VMULPS Y1, Y3, Y3
	VADDPS mathConsts<>+288(SB), Y3, Y3
	VCVTTPS2DQ Y2, Y2
	VPSLLD $23, Y2, Y2
	VPADDD Y2, Y3, Y1
	VMOVUPS Y1, (SI)
	VADDPS  Y1, Y9, Y9
	ADDQ    $32, SI
	DECQ    CX
	JNZ     sloop
	VEXTRACTF128 $1, Y9, X1
	VADDPS  X1, X9, X9
	VSHUFPS $0x4E, X9, X9, X1
	VADDPS  X1, X9, X9
	VSHUFPS $0xB1, X9, X9, X1
	VADDPS  X1, X9, X9
	VMOVSS  X9, ret+24(FP)
	VZEROUPPER
	RET

// func gegluErfAVX2(a, x, gate *float32, n int)
//
// a[i] = gelu_erf(x[i]) * g[i] for n values (n a positive multiple of 8),
// gelu_erf(x) = x/2 · (1 + erf(x/√2)) with erf by A&S 7.1.26.
TEXT ·gegluErfAVX2(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), DI
	MOVQ x+8(FP), SI
	MOVQ gate+16(FP), DX
	MOVQ n+24(FP), CX
	SHRQ $3, CX

gloop:
	VMOVUPS (SI), Y0 // x
	VANDPS  mathConsts<>+640(SB), Y0, Y7 // sign bits
	VXORPS  Y7, Y0, Y1 // |x|
	VMULPS  mathConsts<>+576(SB), Y1, Y1 // u = |x|/√2
	VMULPS  mathConsts<>+384(SB), Y1, Y2
	VADDPS  mathConsts<>+288(SB), Y2, Y2
	VMOVUPS mathConsts<>+288(SB), Y3
	VDIVPS  Y2, Y3, Y2 // t = 1/(1 + p u)
	VMOVUPS mathConsts<>+544(SB), Y3
	VMULPS Y2, Y3, Y3
	VADDPS mathConsts<>+512(SB), Y3, Y3
	VMULPS Y2, Y3, Y3
	VADDPS mathConsts<>+480(SB), Y3, Y3
	VMULPS Y2, Y3, Y3
	VADDPS mathConsts<>+448(SB), Y3, Y3
	VMULPS Y2, Y3, Y3
	VADDPS mathConsts<>+416(SB), Y3, Y3
	VMULPS Y2, Y3, Y3 // poly
	VMULPS Y1, Y1, Y4
	VXORPS mathConsts<>+640(SB), Y4, Y4 // -u²
	VMAXPS mathConsts<>+320(SB), Y4, Y5
	VMINPS mathConsts<>+352(SB), Y5, Y5
	VMULPS mathConsts<>+0(SB), Y5, Y6
	VROUNDPS $0, Y6, Y6
	VMULPS mathConsts<>+32(SB), Y6, Y2
	VSUBPS Y2, Y5, Y5
	VMULPS mathConsts<>+64(SB), Y6, Y2
	VSUBPS Y2, Y5, Y5
	VMOVUPS mathConsts<>+256(SB), Y2
	VMULPS Y5, Y2, Y2
	VADDPS mathConsts<>+224(SB), Y2, Y2
	VMULPS Y5, Y2, Y2
	VADDPS mathConsts<>+192(SB), Y2, Y2
	VMULPS Y5, Y2, Y2
	VADDPS mathConsts<>+160(SB), Y2, Y2
	VMULPS Y5, Y2, Y2
	VADDPS mathConsts<>+128(SB), Y2, Y2
	VMULPS Y5, Y2, Y2
	VADDPS mathConsts<>+96(SB), Y2, Y2
	VMULPS Y5, Y2, Y2
	VADDPS mathConsts<>+288(SB), Y2, Y2
	VMULPS Y5, Y2, Y2
	VADDPS mathConsts<>+288(SB), Y2, Y2
	VCVTTPS2DQ Y6, Y6
	VPSLLD $23, Y6, Y6
	VPADDD Y6, Y2, Y5
	VMULPS  Y5, Y3, Y3
	VMOVUPS mathConsts<>+288(SB), Y6
	VSUBPS  Y3, Y6, Y6 // erf(|x|/√2)
	VORPS   Y7, Y6, Y6 // with the sign of x
	VADDPS  mathConsts<>+288(SB), Y6, Y6
	VMULPS  Y0, Y6, Y6
	VMULPS  mathConsts<>+608(SB), Y6, Y6
	VMULPS  (DX), Y6, Y6
	VMOVUPS Y6, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DX
	ADDQ    $32, DI
	DECQ    CX
	JNZ     gloop
	VZEROUPPER
	RET
