#include <iostream>
#include <string.h>
#include <gc_vdk.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>



extern unsigned char *outputRgbaMem;

// outputRgbaMem = (unsigned char *)calloc(1280 * 720 * 4,sizeof(unsigned char));
extern void glinit(void);                                           // 初始化OpenGL
extern void swap_egl(void);                                         // 使用EGL顯示畫面
extern void imageShow(int width, int height, unsigned char rgb[]);  // OpenGL打畫面