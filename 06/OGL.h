#pragma once
#define MYICON 101

#define MY_EARTH 102

#define MY_STARS 103

#include <windows.h>
#include <iostream>

extern FILE *gpFILE;

extern HWND ghwnd; // global hwnd

extern int gheight;
extern int gwidth;

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_TEXCOORD
};

void uninitialize(void);
