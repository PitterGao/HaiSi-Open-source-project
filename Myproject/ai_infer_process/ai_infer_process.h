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

#ifndef AI_INFER_PROCESS_H
#define AI_INFER_PROCESS_H

#include <stdint.h>
#include "sample_comm_nnie.h"

#if __cplusplus
extern "C" {
#endif

#define HI_PER_BASE         100
#define HI_OVEN_BASE        2 // Even base



/*
 * 与插件有关的信息
 * Plug related information
 */
typedef struct AiPlugLib {
    int width;
    int height;
    uintptr_t model;
} AiPlugLib;


/*
 * 销毁CNN模型
 * Destroy CNN model
 */
void CnnDestroy(SAMPLE_SVP_NNIE_CFG_S *self);

/*
 * 基于模型文件创建CNN模型
 * Create CNN model based mode file
 */
int CnnCreate(SAMPLE_SVP_NNIE_CFG_S **model, const char* modelFile);






#ifdef __cplusplus
}
#endif
#endif
