#pragma once
#include "OGL.h"
#include "vmath.h"
#include <fcntl.h>    // For _O_RDONLY
#include <sys/stat.h> // For _S_IREAD
#include <io.h>
#include <gl/glew.h>
using namespace vmath;
class TextureShader
{
public:
    // Variables
    /* SKY RELATED VARIALBES*/
    GLuint shaderProgramObject;

    GLuint projectionMatrixUniform;
    GLuint viewMatrixUniform;
    GLuint modelMatrixUniform;
    GLuint textureSamplerUniform;
    GLuint alphaValueUniform;
    GLuint vao;
    GLuint vbo_position;
    GLuint vbo_texcoords;

    enum GLShaderType
    {
        VERTEX = 0,
        TESSELLATION_CONTROL,
        TESSELLATION_EVALUATION,
        GEOMETRY,
        FRAGMENT,
        COMPUTE
    };

    BOOL initialize(void)
    {
        // vertex Shader
        GLuint vertexShaderObject = CreateAndCompileShaderObjects("./src/shaders/texture/texture.vs", VERTEX);

        // fragment Shader
        GLuint fragmentShaderObject = CreateAndCompileShaderObjects("./src/shaders/texture/texture.fs", FRAGMENT);

        shaderProgramObject = glCreateProgram();
        glAttachShader(shaderProgramObject, vertexShaderObject);
        glAttachShader(shaderProgramObject, fragmentShaderObject);

        // prelinked binding
        // Binding Position Array
        glBindAttribLocation(shaderProgramObject, AMC_ATTRIBUTE_POSITION, "a_position");
        // Binding Color Array
        glBindAttribLocation(shaderProgramObject, AMC_ATTRIBUTE_TEXCOORD, "a_texcoord");

        // link
        BOOL bShaderLinkStatus = LinkShaderProgramObject(shaderProgramObject);

        if (bShaderLinkStatus == FALSE)
            return FALSE;

        // post link - getting
        projectionMatrixUniform = glGetUniformLocation(shaderProgramObject, "u_projectionMatrix");
        viewMatrixUniform = glGetUniformLocation(shaderProgramObject, "u_viewMatrix");
        modelMatrixUniform = glGetUniformLocation(shaderProgramObject, "u_modelMatrix");
        textureSamplerUniform = glGetUniformLocation(shaderProgramObject, "u_textureSampler");
        alphaValueUniform = glGetUniformLocation(shaderProgramObject, "u_alphaVal");

        // VAO For QUAD

        const GLfloat positions[] =
            {
                // front
                1.0f, 1.0f, 0.0f,
                -1.0f, 1.0f, 0.0f,
                -1.0f, -1.0f, 0.0f,
                1.0f, -1.0f, 0.0f};

        const GLfloat texcoord[] =
            {
                1.0, 0.0,
                0.0, 0.0,
                0.0, 1.0,
                1.0, 1.0};

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        {
            glGenBuffers(1, &vbo_position);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
            glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glGenBuffers(1, &vbo_texcoords);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords);
            glBufferData(GL_ARRAY_BUFFER, sizeof(texcoord), texcoord, GL_STATIC_DRAW);
            glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        glBindVertexArray(0);

        return TRUE;
    }

    void drawQuadWithTexture(GLuint texture, mat4 modelMatrix, mat4 viewMatrix, mat4 perspectiveProjectionMatrix, float alpha)
    {
        glUseProgram(shaderProgramObject);
        {
            // Uniforms
            glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
            glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, viewMatrix);
            glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix);
            glUniform1f(alphaValueUniform, alpha);
            glUniform1i(textureSamplerUniform, 0);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4); // 4 Vertices for cube
            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glDisable(GL_BLEND);
        }
        glUseProgram(0);
    }
    void uninitialize(void)
    {
        UninitializeShaders(shaderProgramObject);
    }

    GLuint CreateAndCompileShaderObjects(const char *shaderFilename, GLShaderType shaderType)
    {
        // variable declarations
        GLuint shaderObject = 0;
        char shaderNameTag[50];
        GLenum glShaderType;

        int fdShaderFile = -1;
        long shaderFileSize = -1;
        char *shaderSourceCodeBuffer = NULL;

        int status = 0;
        int infoLogLength = 0;
        char *log = NULL;

        // code
        fdShaderFile = _open(shaderFilename, _O_RDONLY, _S_IREAD);
        if (fdShaderFile == -1)
        {
            fprintf(gpFILE, "Failed To Open Shader File %s! Exitting Now ...\n", shaderFilename);
            return (0);
        }

        shaderFileSize = _lseek(fdShaderFile, 0, SEEK_END);
        shaderFileSize = shaderFileSize + 1; // '\0'
        shaderSourceCodeBuffer = (char *)malloc(shaderFileSize);
        _lseek(fdShaderFile, 0, SEEK_SET);

        shaderFileSize = _read(fdShaderFile, (char *)shaderSourceCodeBuffer, shaderFileSize);
        if (shaderFileSize < 0)
        {
            fprintf(gpFILE, "Failed To Read Shader File %s! Exitting Now ...\n", shaderFilename);
            return (0);
        }

        shaderSourceCodeBuffer[shaderFileSize] = '\0';
        _close(fdShaderFile);

        switch (shaderType)
        {
        case VERTEX:
            strcpy_s(shaderNameTag, 50, "Vertex");
            glShaderType = GL_VERTEX_SHADER;
            break;

        case TESSELLATION_CONTROL:
            strcpy_s(shaderNameTag, 50, "Tessellation Control");
            glShaderType = GL_TESS_CONTROL_SHADER;
            break;

        case TESSELLATION_EVALUATION:
            strcpy_s(shaderNameTag, 50, "Tessellation Evaluation");
            glShaderType = GL_TESS_EVALUATION_SHADER;
            break;

        case GEOMETRY:
            strcpy_s(shaderNameTag, 50, "Geometry");
            glShaderType = GL_GEOMETRY_SHADER;
            break;

        case FRAGMENT:
            strcpy_s(shaderNameTag, 50, "Fragment");
            glShaderType = GL_FRAGMENT_SHADER;
            break;

        case COMPUTE:
            strcpy_s(shaderNameTag, 50, "Compute");
            glShaderType = GL_COMPUTE_SHADER;
            break;

        default:
            fprintf(gpFILE, "Invalid Shader Type! Exitting Now ...\n");
            return (0);
        }

        // CREATING SHADER OBJECT
        shaderObject = glCreateShader(glShaderType);
        if (shaderObject == 0)
        {
            fprintf(gpFILE, "\"%s\" : Failed To Create %s Shader Object! Exitting Now ...\n", shaderFilename, shaderNameTag);
            return (0);
        }

        glShaderSource(shaderObject, 1, (const GLchar **)&shaderSourceCodeBuffer, NULL);
        free(shaderSourceCodeBuffer);

        glCompileShader(shaderObject);

        // Step 5

        // 5(a)
        glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &status);
        if (status == GL_FALSE)
        {
            fprintf(gpFILE, "\"%s\" : %s Shader Compilation Failed.\n", shaderFilename, shaderNameTag);
            glGetShaderiv(shaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
            if (infoLogLength > 0)
            {
                log = (char *)malloc(sizeof(char) * infoLogLength);
                if (log != NULL)
                {
                    GLsizei written;
                    glGetShaderInfoLog(shaderObject, infoLogLength, &written, log);
                    fprintf(gpFILE, "\"%s\" : %s Shader Compilation Log : %s\n", shaderFilename, shaderNameTag, log);
                    free(log);
                    return (0);
                }
            }
        }

        else
        {
            fprintf(gpFILE, "\"%s\" : %s Shader Compilation Succeeded.\n", shaderFilename, shaderNameTag);
        }

        return (shaderObject);
    }

    BOOL LinkShaderProgramObject(GLuint shaderProgramObject)
    {
        // variable declarations
        int status = 0;
        int infoLogLength = 0;
        char *log = NULL;

        // code
        glLinkProgram(shaderProgramObject);

        // Step 4
        glGetProgramiv(shaderProgramObject, GL_LINK_STATUS, &status);
        if (status == GL_FALSE)
        {
            fprintf(gpFILE, "Shader Program Linking Failed.\n");
            glGetProgramiv(shaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);
            if (infoLogLength > 0)
            {
                log = (char *)malloc(sizeof(char) * infoLogLength);
                if (log != NULL)
                {
                    GLsizei written;
                    glGetProgramInfoLog(shaderProgramObject, infoLogLength, &written, log);
                    fprintf(gpFILE, "Shader Program Linking Log : %s\n", log);
                    free(log);
                    return (FALSE);
                }
            }
        }

        else
        {
            fprintf(gpFILE, "Shader Program Linking Succeeded.\n");
        }

        return (TRUE);
    }

    void UninitializeShaders(GLuint shaderProgramObject)
    {
        // code
        if (shaderProgramObject)
        {
            glUseProgram(shaderProgramObject);

            GLsizei numAttachedShaders = 0;
            glGetProgramiv(shaderProgramObject, GL_ATTACHED_SHADERS, &numAttachedShaders);

            GLuint *shaderObjects = NULL;
            shaderObjects = (GLuint *)malloc(sizeof(GLuint) * numAttachedShaders);

            glGetAttachedShaders(shaderProgramObject, numAttachedShaders, &numAttachedShaders, shaderObjects);

            for (GLsizei i = 0; i < numAttachedShaders; i++)
            {
                glDetachShader(shaderProgramObject, shaderObjects[i]);
                glDeleteShader(shaderObjects[i]);
                shaderObjects[i] = 0;
            }

            free(shaderObjects);
            shaderObjects = NULL;

            glUseProgram(0);

            glDeleteProgram(shaderProgramObject);
            shaderProgramObject = 0;
        }
    }
};
