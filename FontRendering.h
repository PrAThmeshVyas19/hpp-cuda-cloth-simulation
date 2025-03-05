#ifndef FONT_RENDERING_H
#define FONT_RENDERING_H

#include <map>
#include <string>
#include <GL/glew.h>
#include "vmath.h"
#include <ft2build.h>
#include FT_FREETYPE_H

#include "OGL.h"

class FontRendering
{
public:
	FontRendering();
	~FontRendering();

	int initFontRendering();
	void renderText(std::string text, float x, float y, float scale, float textColor[], vmath::mat4 projectionMatrix);

private:
	struct Character
	{
		GLuint textureID;
		int size[2];
		int bearing[2];
		GLuint advance;
	};

public:
	std::map<char, Character> characters;
	GLuint gShaderProgramObjectFontRender;
	GLuint vaoFontRender, vboFontRender;
	GLuint textTextureSampleUniformFontRender, mvpUniformFontRender, textColorUniform;
};

#endif // FONT_RENDERING_H
