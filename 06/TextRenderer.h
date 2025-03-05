#pragma once

// header files
#include "OGL.h"

// frree type header
#include <ft2build.h>
#include FT_FREETYPE_H
#include <gl/glew.h>
#include "vmath.h"

// funtion declarationns
void initializeTextRenderers(const char *fontPath);
void drawStringTextRenderer(const char *string, vmath::vec2 position, vmath::vec2 scale, vmath::vec3 color, vmath::mat4 projectionMatrix);
void uninitializerenderer(void);
