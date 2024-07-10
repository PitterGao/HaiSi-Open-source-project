#ifndef TEXT_0_H
#define TEXT_0_H
#include <iostream>
#include "sample_comm_nnie.h"
#ifdef __cplusplus

extern "C" {
#endif
typedef struct tagIPC_IMAGE {
    HI_U64 u64PhyAddr;
    HI_U64 u64VirAddr;
    HI_U32 u32Width;
    HI_U32 u32Height;
} IPC_IMAGE;
static HI_VOID IveImageParamCfg(IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
    VIDEO_FRAME_INFO_S *srcFrame);

static HI_S32 yuvFrame2rgb(VIDEO_FRAME_INFO_S *srcFrame, IPC_IMAGE *dstImage);

HI_S32 save_background(VIDEO_FRAME_INFO_S*src);

static HI_S32 frame2Mat(VIDEO_FRAME_INFO_S *srcFrame);
#ifdef __cplusplus
}
#endif

#endif  // YOUR_CPP_FUNCTIONS_HPP