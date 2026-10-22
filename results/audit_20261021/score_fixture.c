#include "fault_inject.h"
#include <stdio.h>
#include <string.h>
static int image_index, failure_phase;
TfLiteTensor *TfLiteInterpreterGetInputTensor(const TfLiteInterpreter *i, int n) {(void)i;(void)n;return NULL;}
const TfLiteTensor *TfLiteInterpreterGetOutputTensor(const TfLiteInterpreter *i, int n) {(void)i;(void)n;return NULL;}
TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor *t,const void *p,size_t n) {
    (void)t;(void)n;image_index=*(const int8_t *)p;
    return image_index==3 && failure_phase==0 ? kTfLiteError:kTfLiteOk;
}
TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter *i) {
    (void)i;return image_index==3 && failure_phase==1 ? kTfLiteError:kTfLiteOk;
}
TfLiteStatus TfLiteTensorCopyToBuffer(const TfLiteTensor *t,void *p,size_t n) {
    (void)t;memset(p,0,n);int8_t *v=p;v[0]=10;
    if(image_index==1)v[1]=1;
    if(image_index==2)v[2]=20;
    return image_index==3 && failure_phase==2 ? kTfLiteError:kTfLiteOk;
}
int main(void) {
    int8_t images[]={0,1,2,3},clean[4*FI_NUM_CLASSES]={0};int32_t labels[]={0,0,2,0};
    for(int j=0;j<4;j++)clean[j*FI_NUM_CLASSES]=10;
    fi_runner_t runner={0}; runner.in_bytes=1;
    fi_subset_t subset={4,1,images,labels};
    for(failure_phase=0;failure_phase<3;failure_phase++) {
        fi_score_t score;fi_score_run(&runner,&subset,clean,&score);
        if(score.n_masked!=1||score.n_subtle!=1||score.n_catastrophic!=1||score.n_due!=1||score.n_correct_clean!=3||score.n_correct_faulted!=3)return 1;
    }
    int8_t ties[FI_NUM_CLASSES];memset(ties,-128,sizeof ties);ties[5]=127;ties[9]=127;
    if(fi_argmax(ties)!=5)return 2;
    memset(ties,-128,sizeof ties);if(fi_argmax(ties)!=0)return 3;
    puts("pass: known masked/subtle/catastrophic counts, all three returned-error paths, accuracy accounting, first-maximum ties");
    return 0;
}
