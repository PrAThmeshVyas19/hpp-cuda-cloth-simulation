// header files
#include "TextRenderer.h"

// struct
typedef struct Character
{
    GLuint textureID;     // ID of the glyph texture
    vmath::uvec2 size;    // size of glyph
    vmath::ivec2 bearing; // offset from baseline to left/top of glyph
    unsigned int advance; // offset to advance to next glyph
} Character;

// global variables
static GLuint g_shaderProgramObject = 0;
static GLuint projectionMatrixUniform = 0;
static GLuint g_textureSamplerUniform = 0;
static GLuint g_mvpMatrixUniform = 0;
static GLuint colorUniform = 0;

// VAOs and VBOs
static GLuint g_skyboxVAO = 0;
static GLuint g_skyboxVBO = 0;
static GLuint vao = 0;
static GLuint vbo = 0;

static GLuint asciiTextures[128];
static Character characterArray[128];

// function definitions
void initializeTextRenderers(const char *fontPath)
{
    void CreateFontTextures(const char *fontPath);

    // vertex shader
    const GLchar *vertexShaderSourceCode =
        "#version 460 core\n"
        "\n"
        "in vec4 aPosition;\n"
        "in vec2 aTexCoord;\n"
        "\n"
        "uniform mat4 uProjectionMatrix;\n"
        "\n"
        "out vec2 oTexCoord;\n"
        "\n"
        "void main(void)\n"
        "{\n"
        "    gl_Position = uProjectionMatrix * aPosition;\n"
        "    oTexCoord   = aTexCoord;\n"
        "}\n";

    // create shader object
    GLuint vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

    // give the shader source code to the OpenGL server
    glShaderSource(vertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, nullptr);

    // compile the shader
    glCompileShader(vertexShaderObject);

    // check for any compilation errors
    GLint status = 0;
    GLint infoLogLength = 0;
    GLchar *szInfoLog = nullptr;

    glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &status);

    if (status == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(sizeof(GLchar) * (infoLogLength + 1));

            if (szInfoLog != nullptr)
            {
                glGetShaderInfoLog(vertexShaderObject, infoLogLength + 1, nullptr, szInfoLog);

                fprintf(gpFILE, "initializeSkybox() : Vertex Shader compilation error info log : \n%s\n", szInfoLog);

                free(szInfoLog);
                szInfoLog = nullptr;
            }
        }

        uninitialize();
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(gpFILE, "initializeSkybox() : Vertex Shader was compiled successfully.\n");
    }

    // fragment shader
    const GLchar *fragmentShaderSourceCode =
        "#version 460 core\n"
        "\n"
        "in vec2 oTexCoord;\n"
        "\n"
        "uniform sampler2D uTextureSampler;\n"
        "uniform vec3 uColor; \n"
        "out vec4 FragColor;\n"
        "\n"
        "void main(void)\n"
        "{\n"
        "vec4 alpha = vec4(1.0, 1.0, 1.0, texture(uTextureSampler, oTexCoord).r);"
        "FragColor = vec4(uColor, 1.0) * alpha;"
        "}\n";

    // create shader object
    GLuint fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

    // give the shader source code to the OpenGL server
    glShaderSource(fragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, nullptr);

    // compile the shader
    glCompileShader(fragmentShaderObject);

    // check for any compilation errors
    status = 0;
    infoLogLength = 0;
    szInfoLog = nullptr;

    glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &status);

    if (status == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(sizeof(GLchar) * (infoLogLength + 1));

            if (szInfoLog != nullptr)
            {
                glGetShaderInfoLog(fragmentShaderObject, infoLogLength + 1, nullptr, szInfoLog);

                fprintf(gpFILE, "initializeSkybox() : Fragment Shader compilation error info log : \n%s\n", szInfoLog);

                free(szInfoLog);
                szInfoLog = nullptr;
            }
        }

        uninitialize();
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(gpFILE, "initializeSkybox() : Fragment Shader was compiled successfully.\n");
    }

    // ************** SHADER PROGRAM **************
    // create the shader program
    g_shaderProgramObject = glCreateProgram();

    // attach the shaders
    glAttachShader(g_shaderProgramObject, vertexShaderObject);
    glAttachShader(g_shaderProgramObject, fragmentShaderObject);

    // bind the vertex attributes
    glBindAttribLocation(g_shaderProgramObject, AMC_ATTRIBUTE_POSITION, "aPosition");
    glBindAttribLocation(g_shaderProgramObject, AMC_ATTRIBUTE_TEXCOORD, "aTexCoord");

    // link the program
    glLinkProgram(g_shaderProgramObject);

    // check for any compilation errors
    status = 0;
    infoLogLength = 0;
    szInfoLog = nullptr;

    glGetProgramiv(g_shaderProgramObject, GL_LINK_STATUS, &status);

    if (status == GL_FALSE)
    {
        glGetProgramiv(g_shaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);

        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(sizeof(GLchar) * (infoLogLength + 1));

            if (szInfoLog != nullptr)
            {
                glGetProgramInfoLog(g_shaderProgramObject, infoLogLength + 1, nullptr, szInfoLog);

                fprintf(gpFILE, "initializeSkybox() : Shader program linkage error info log : \n%s\n", szInfoLog);

                free(szInfoLog);
                szInfoLog = nullptr;
            }
        }

        uninitialize();
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(gpFILE, "initializeSkybox() : Shader program was linked successfully.\n");
    }

    // ************** GET UNIFORM LOCATIONS **************
    projectionMatrixUniform = glGetUniformLocation(g_shaderProgramObject, "uProjectionMatrix");
    g_textureSamplerUniform = glGetUniformLocation(g_shaderProgramObject, "uTextureSampler");
    colorUniform = glGetUniformLocation(g_shaderProgramObject, "uColor");

    // vao and vbo
    glGenVertexArrays(1, &vao);

    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4, NULL);
    glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

    glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4, (const void *)(sizeof(GLfloat) * 2));
    glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    // call
    CreateFontTextures(fontPath);
}

void drawStringTextRenderer(const char *string, vmath::vec2 position, vmath::vec2 scale, vmath::vec3 color, vmath::mat4 projectionMatrix)
{
    // code
    // set opengl state
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(g_shaderProgramObject);
    glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, projectionMatrix);
    glUniform3fv(colorUniform, 1, color);
    glBindVertexArray(vao);
    // loop through all char in the string
    size_t stringLength = strlen(string);
    for (size_t i = 0; i < stringLength; i++)
    {
        Character currentCharacter = characterArray[string[i]];

        // calculate x pos and y pos of array
        GLfloat x = position[0] + (GLfloat)currentCharacter.bearing[0] * scale[0];
        GLfloat y = position[1] - (GLfloat)(currentCharacter.size[1] - currentCharacter.bearing[1]) * scale[1];

        // calculate wiidth and height of char
        GLfloat width = currentCharacter.size[0] * scale[0];
        GLfloat height = currentCharacter.size[1] * scale[1];

        // update vbo
        GLfloat quadData[6][4] =
            {
                {x, y + height, 0.0f, 0.0f},
                {x, y, 0.0f, 1.0f},
                {x + width, y, 1.0f, 1.0f},

                {x, y + height, 0.0f, 0.0f},
                {x + width, y, 1.0f, 1.0f},
                {x + width, y + height, 1.0f, 0.0f},
            };

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, quadData, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentCharacter.textureID);
        glUniform1i(g_textureSamplerUniform, 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindTexture(GL_TEXTURE_2D, 0);

        // advc the cursor fot the next glyph (keep cursor ahead of the prev later)
        position[0] = position[0] + (currentCharacter.advance >> 6) * scale[0];
    }

    glBindVertexArray(0);
    glUseProgram(0);

    // rest the opengl state
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void uninitializerenderer(void)
{
    glDeleteTextures(128, asciiTextures);

    // delete Skybox VBO
    if (g_skyboxVBO)
    {
        glDeleteBuffers(1, &g_skyboxVBO);
        g_skyboxVBO = 0;
    }

    // delete Skybox VAO
    if (g_skyboxVAO)
    {
        glDeleteVertexArrays(1, &g_skyboxVAO);
        g_skyboxVAO = 0;
    }

    // delete Skybox VBO
    if (vao)
    {
        glDeleteBuffers(1, &vao);
        vao = 0;
    }

    // delete Skybox VAO
    if (vbo)
    {
        glDeleteVertexArrays(1, &vbo);
        vbo = 0;
    }

    // delete the shader program
    if (g_shaderProgramObject)
    {
        glUseProgram(g_shaderProgramObject);
        {
            GLint numShaders = 0;
            glGetProgramiv(g_shaderProgramObject, GL_ATTACHED_SHADERS, &numShaders);

            if (numShaders > 0)
            {
                GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * numShaders);
                if (pShaders != nullptr)
                {
                    glGetAttachedShaders(g_shaderProgramObject, numShaders, nullptr, pShaders);

                    for (int i = 0; i < numShaders; i++)
                    {
                        glDetachShader(g_shaderProgramObject, pShaders[i]);
                        glDeleteShader(pShaders[i]);
                        pShaders[i] = 0;
                    }

                    free(pShaders);
                    pShaders = nullptr;
                }
            }
        }
        glUseProgram(0);

        glDeleteProgram(g_shaderProgramObject);
        g_shaderProgramObject = 0;
    }
}

// internal functions
void CreateFontTextures(const char *fontPath)
{
    FT_Library ft;
    FT_Face font;

    // code
    // init the freetype llibrary
    if (FT_Init_FreeType(&ft))
    {
        fprintf(gpFILE, "FT_init_free_type failed\n");
        uninitialize();
        exit(EXIT_FAILURE);
    }

    // initialize the font
    if (FT_New_Face(ft, fontPath, 0, &font))
    {
        fprintf(gpFILE, "FT_New_Face failed to load %s font\n", fontPath);
        uninitialize();
        exit(EXIT_FAILURE);
    }

    // set size of the font
    FT_Set_Pixel_Sizes(font, 0, 32); // fnont, width, height
    // internally height nusar wodth adjust karnar

    // load all ascii characters and store them for later use
    int index = 0;
    for (unsigned char c = 0; c < 128; c++)
    {
        // load Character glyph
        if (FT_Load_Char(font, c, FT_LOAD_RENDER))
        {
            fprintf(gpFILE, "FT_Load_Char failed\n");
            uninitialize();
            exit(EXIT_FAILURE);
        }

        GLuint texture;
        glGenTextures(1, &texture);
        asciiTextures[index] = texture;

        index++;
        glBindTexture(GL_TEXTURE_2D, texture);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            font->glyph->bitmap.width,
            font->glyph->bitmap.rows,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            font->glyph->bitmap.buffer);

        glBindTexture(GL_TEXTURE_2D, 0);

        // now store character for later use
        characterArray[c].textureID = texture;
        characterArray[c].size = vmath::uvec2(font->glyph->bitmap.width, font->glyph->bitmap.rows);
        characterArray[c].bearing = vmath::ivec2(font->glyph->bitmap_left, font->glyph->bitmap_top);
        characterArray[c].advance = font->glyph->advance.x;
    }

    // claen up freetype
    FT_Done_Face(font);
    FT_Done_FreeType(ft);
}
