#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "save_background.h"
#include "sample_comm_ive.h"

using namespace std;
using namespace cv;
static IVE_SRC_IMAGE_S pstSrc;
static IVE_DST_IMAGE_S pstDst;
static IVE_CSC_CTRL_S stCscCtrl;
extern "C"{

static HI_VOID IveImageParamCfg(IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
    VIDEO_FRAME_INFO_S *srcFrame)
{
    pstSrc->enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrc->au64VirAddr[0] = srcFrame->stVFrame.u64VirAddr[0];
    pstSrc->au64VirAddr[1] = srcFrame->stVFrame.u64VirAddr[1];
    pstSrc->au64VirAddr[2] = srcFrame->stVFrame.u64VirAddr[2]; // 2: Image data virtual address

    pstSrc->au64PhyAddr[0] = srcFrame->stVFrame.u64PhyAddr[0];
    pstSrc->au64PhyAddr[1] = srcFrame->stVFrame.u64PhyAddr[1];
    pstSrc->au64PhyAddr[2] = srcFrame->stVFrame.u64PhyAddr[2]; // 2: Image data physical address

    pstSrc->au32Stride[0] = srcFrame->stVFrame.u32Stride[0];
    pstSrc->au32Stride[1] = srcFrame->stVFrame.u32Stride[1];
    pstSrc->au32Stride[2] = srcFrame->stVFrame.u32Stride[2]; // 2: Image data span

    pstSrc->u32Width = srcFrame->stVFrame.u32Width;
    pstSrc->u32Height = srcFrame->stVFrame.u32Height;

    pstDst->enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
    pstDst->u32Width = pstSrc->u32Width;
    pstDst->u32Height = pstSrc->u32Height;
    pstDst->au32Stride[0] = pstSrc->au32Stride[0];
    pstDst->au32Stride[1] = 0;
    pstDst->au32Stride[2] = 0; // 2: Image data span
}

static HI_S32 yuvFrame2rgb(VIDEO_FRAME_INFO_S *srcFrame, IPC_IMAGE *dstImage)
{
    IVE_HANDLE hIveHandle;
    HI_S32 s32Ret = 0;
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2RGB; // IVE_CSC_MODE_VIDEO_BT601_YUV2RGB
    IveImageParamCfg(&pstSrc, &pstDst, srcFrame);

    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(&pstDst.au64PhyAddr[0], (void **)&pstDst.au64VirAddr[0],
        "User", HI_NULL, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        SAMPLE_PRT("HI_MPI_SYS_MmzFree err\n");
        return s32Ret;
    }

    s32Ret = HI_MPI_SYS_MmzFlushCache(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0],
        pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }
    // 3: multiple
    memset_s((void *)pstDst.au64VirAddr[0], pstDst.u32Height*pstDst.au32Stride[0] * 3,
        0, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    HI_BOOL bInstant = HI_TRUE;

    s32Ret = HI_MPI_IVE_CSC(&hIveHandle, &pstSrc, &pstDst, &stCscCtrl, bInstant);
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }

    if (HI_TRUE == bInstant) {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret) {
            usleep(100); // 100: usleep time
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
    }
    dstImage->u64PhyAddr = pstDst.au64PhyAddr[0];
    dstImage->u64VirAddr = pstDst.au64VirAddr[0];
    dstImage->u32Width = pstDst.u32Width;
    dstImage->u32Height = pstDst.u32Height;

    return HI_SUCCESS;
}

static HI_S32 frame2Mat(VIDEO_FRAME_INFO_S *srcFrame)
{
    HI_U32 w = srcFrame->stVFrame.u32Width;
    HI_U32 h = srcFrame->stVFrame.u32Height;
    Mat dstMat; // 声明一个Mat对象
    int bufLen = w * h * 3;
    HI_U8 *srcRGB = NULL;
    IPC_IMAGE dstImage;
    if (yuvFrame2rgb(srcFrame, &dstImage) != HI_SUCCESS) {
        SAMPLE_PRT("yuvFrame2rgb err\n");
        return HI_FAILURE;
    }
    srcRGB = (HI_U8 *)dstImage.u64VirAddr;
    dstMat.create(h, w, CV_8UC3);
    memcpy_s(dstMat.data, bufLen * sizeof(HI_U8), srcRGB, bufLen * sizeof(HI_U8));
 cout<<"GetSegnetMap success"<<endl; 
    Mat resizedMatt;
    resize(dstMat, resizedMatt, Size(1920, 1080), 0, 0, INTER_NEAREST);
    cout<<"resize success"<<endl;
   bool success = imwrite("background.bmp", resizedMatt);
if (success) {
        std::cout << "successed" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
    cout<<"imwrite success"<<endl;
    dstMat.release();
    HI_MPI_SYS_MmzFree(dstImage.u64PhyAddr, (void *)&(dstImage.u64VirAddr));
    return HI_SUCCESS;
}

HI_S32 save_background(VIDEO_FRAME_INFO_S*src)
{   HI_S32 ret;
    ret=frame2Mat(src);
    return ret;
}
}
