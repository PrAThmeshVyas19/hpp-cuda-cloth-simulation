#include "FontRendering.h"
#include <iostream>

FontRendering::FontRendering()
{
	gShaderProgramObjectFontRender = 0;
	vaoFontRender = 0;
	vboFontRender = 0;
}

FontRendering::~FontRendering()
{
	glDeleteBuffers(1, &vboFontRender);
	glDeleteVertexArrays(1, &vaoFontRender);
	glDeleteProgram(gShaderProgramObjectFontRender);
}

// Function to check shader compile errors
void checkShaderCompile(GLuint shader)
{
	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		GLchar infoLog[1024];
		glGetShaderInfoLog(shader, 1024, NULL, infoLog);
		fprintf(gpFILE, "ERROR: Shader Compilation Failed\n%s\n", infoLog);
	}
}

// Function to check shader program link errors
void checkProgramLink(GLuint program)
{
	GLint success;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success)
	{
		GLchar infoLog[1024];
		glGetProgramInfoLog(program, 1024, NULL, infoLog);
		fprintf(gpFILE, "ERROR: Program Linking Failed\n%s\n", infoLog);
	}
}

int FontRendering::initFontRendering()
{
	FT_Library ft;
	if (FT_Init_FreeType(&ft))
	{
		fprintf(gpFILE, "ERROR: Could not initialize FreeType library\n");
		return -1;
	}

	FT_Face face;
	if (FT_New_Face(ft, "coolvetica.otf", 0, &face))
	{
		fprintf(gpFILE, "ERROR: Failed to load font coolvetica.otf\n");
		return -1;
	}

	FT_Set_Pixel_Sizes(face, 0, 32);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	for (unsigned char c = 0; c < 128; c++)
	{
		if (FT_Load_Char(face, c, FT_LOAD_RENDER))
		{
			fprintf(gpFILE, "ERROR: Failed to load character\n");
			continue;
		}

		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows,
					 0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		Character character = {
			texture,
			{face->glyph->bitmap.width, face->glyph->bitmap.rows},
			{face->glyph->bitmap_left, face->glyph->bitmap_top},
			face->glyph->advance.x};

		characters.insert(std::pair<char, Character>(c, character));
	}

	FT_Done_Face(face);
	FT_Done_FreeType(ft);

	glGenVertexArrays(1, &vaoFontRender);
	glGenBuffers(1, &vboFontRender);
	glBindVertexArray(vaoFontRender);
	glBindBuffer(GL_ARRAY_BUFFER, vboFontRender);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	const char *vertexShaderSource = R"(
        #version 410 core
        layout (location = 0) in vec4 vPosition;
        out vec2 outTexCoord;
        uniform mat4 u_mvpMatrix;
        void main()
        {
            gl_Position = u_mvpMatrix * vec4(vPosition.xy, 0.0, 1.0);
            outTexCoord = vPosition.zw;
        }
    )";
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	checkShaderCompile(vertexShader);

	const char *fragmentShaderSource = R"(
        #version 410 core
        in vec2 outTexCoord;
        out vec4 FragColor;
        uniform sampler2D textSampler;
        uniform vec3 textColor;
        void main()
        {
            float alpha = texture(textSampler, outTexCoord).r;
            FragColor = vec4(textColor, alpha);
        }
    )";
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	checkShaderCompile(fragmentShader);

	gShaderProgramObjectFontRender = glCreateProgram();
	glAttachShader(gShaderProgramObjectFontRender, vertexShader);
	glAttachShader(gShaderProgramObjectFontRender, fragmentShader);
	glLinkProgram(gShaderProgramObjectFontRender);
	checkProgramLink(gShaderProgramObjectFontRender);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	mvpUniformFontRender = glGetUniformLocation(gShaderProgramObjectFontRender, "u_mvpMatrix");
	textTextureSampleUniformFontRender = glGetUniformLocation(gShaderProgramObjectFontRender, "textSampler");
	textColorUniform = glGetUniformLocation(gShaderProgramObjectFontRender, "textColor");

	return 0;
}

void FontRendering::renderText(std::string text, float x, float y, float scale, float textColor[], vmath::mat4 projectionMatrix)
{
	glUseProgram(gShaderProgramObjectFontRender);
	glUniform3fv(textColorUniform, 1, textColor);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);

	glActiveTexture(GL_TEXTURE0);
	glUniform1i(textTextureSampleUniformFontRender, 0);

	glBindVertexArray(vaoFontRender);

	for (char c : text)
	{
		Character ch = characters[c];

		float xPos = x + ch.bearing[0] * scale;
		float yPos = y - (ch.size[1] - ch.bearing[1]) * scale;
		float w = ch.size[0] * scale;
		float h = ch.size[1] * scale;

		float vertices[6][4] = {
			{xPos, yPos + h, 0.0f, 0.0f}, {xPos, yPos, 0.0f, 1.0f}, {xPos + w, yPos, 1.0f, 1.0f}, {xPos, yPos + h, 0.0f, 0.0f}, {xPos + w, yPos, 1.0f, 1.0f}, {xPos + w, yPos + h, 1.0f, 0.0f}};

		glBindTexture(GL_TEXTURE_2D, ch.textureID);
		fprintf(gpFILE, "Text Vertex Data: %f , %f , %f , %f \n", xPos, yPos, w, h);

		glBindBuffer(GL_ARRAY_BUFFER, vboFontRender);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		x += (ch.advance >> 6) * scale;
	}

	glBindVertexArray(0);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glUseProgram(0);
}
