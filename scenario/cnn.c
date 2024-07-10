/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 本文件将垃圾分类wk模型部署到板端，通过NNIE硬件加速进行推理。该文件提供了垃圾分类场景的API接口，
 * 包括模型的加载、模型的卸载、模型的推理、AI flag业务处理接口。支持语音实时播放功能。
 *
 * This file deploys the trash classification wk model to the board,
 * and performs inference through NNIE hardware acceleration.
 * This file provides API interfaces for trash classification scenarios,
 * including model loading, model unloading, model reasoning,
 * and AI flag business processing interfaces. Support audio real-time playback function.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/prctl.h>

#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "posix_help.h"
#include "base_interface.h"
#include "cnn.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define MODEL_FILE_TRASH    "./models/inst_segnet_cycle.wk" // Open source model conversion
#define SCORE_MAX           4096    // The score corresponding to the maximum probability
#define DETECT_OBJ_MAX      32
#define RET_NUM_MAX         4
#define THRESH_MIN          30      // Acceptable probability threshold (over this value will be returned to the app)

#define FRM_WIDTH           224
#define FRM_HEIGHT          224
#define TXT_BEGX            20
#define TXT_BEGY            20


#define MULTIPLE_OF_EXPANSION 100   // Multiple of expansion
#define UNKOWN_WASTE          20    // Unkown Waste
#define BUFFER_SIZE           16    // buffer size
#define MIN_OF_BOX            16    // min of box
#define MAX_OF_BOX            240   // max of box



HI_S32 CnnLoadModel(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;
    ret = CnnCreate(&self, MODEL_FILE_TRASH);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("load cnn trash classify model, ret:%d\n", ret);
    return ret;
}

HI_S32 CnnUnloadModel(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    SAMPLE_PRT("unload trash classify model success\n");
    return HI_SUCCESS;
}

/*
 * 先进行预处理，再使用NNIE进行硬件加速推理，不支持层通过AI CPU进行计算
 *
 * Perform preprocessing first, and then use NNIE for hardware accelerated inference,
 * and do not support layers to be calculated by AI CPU
 */
HI_S32 CnnCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *resFrm)
{
    SAMPLE_PRT("begin CnnAIprocess\n");
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model; // reference to SDK sample_comm_nnie.h Line 99
    IVE_IMAGE_S img; // referece to SDK hi_comm_ive.h Line 143
    VIDEO_FRAME_INFO_S resizeFrm;  // Meet the input frame of the plug

    //RecogNumInfo resBuf[RET_NUM_MAX] = {0};
    //HI_S32 resLen = 0;
    HI_S32 ret;

    ret = MppFrmResize(srcFrm, &resizeFrm, FRM_WIDTH, FRM_HEIGHT);  // resize 224*224
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "for resize FAIL, ret=%x\n", ret);

    ret = save_background(&resizeFrm);

    ret = FrmToRgbImg(&resizeFrm, &img);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != ret, CnnCal_FAIL, "Error(%#x), FrmToRgbImg failed!\n", ret);

    ret = ImgRgbToBgr(&img);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != ret, CnnCal_FAIL, "Error(%#x), ImgRgbToBgr failed!\n", ret);
    
    ret = CnnCalImg_t(self,&img);
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "cnn cal FAIL, ret=%x\n", ret);

    ret = (update_regions("./image.bmp", 8));
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != ret, CnnCal_FAIL, "Error(%#x), update_regions failed!\n", ret);

    MppFrmDestroy(&resizeFrm);


    return HI_SUCCESS;
CnnCal_FAIL:

    return HI_FAILURE;

}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
