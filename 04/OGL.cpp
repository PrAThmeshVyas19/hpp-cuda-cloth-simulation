// common header files
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// OpenGL headers
#include <gl/glew.h>
#include <gl/Gl.h>
#include "vmath.h" // Vector math library
using namespace vmath;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Link OpenGL libraries
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")

// Macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_TEXCOORD
};

GLuint vbo_texcoord = 0;
GLuint textureSamplerUniform = 0;
GLuint Texture1;

// Global variables
HWND ghHWND = NULL;
BOOL gbActive = FALSE;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
GLuint shaderProgramObject;
GLuint vao, vbo;
GLuint ebo;

// Wave Parameters
float waveAmplitude = 0.1f;
float waveFrequency = 1.0f;
float waveSpeed = 2.0f;

// Cloth Grid
const int POINTS_X = 1000;
const int POINTS_Y = 1000;
const int POINTS_TOTAL = (POINTS_X * POINTS_Y);

// Time tracking
std::chrono::time_point<std::chrono::steady_clock> startTime;

// Function declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void initialize(void);
void display(void);
void update(void);
void uninitialize(void);

// Entry point function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    void update(void);

    WNDCLASSEX wndclass;
    HWND hwnd;
    MSG msg;
    BOOL bDone = FALSE;

    // Register Window Class
    wndclass.cbSize = sizeof(WNDCLASSEX);
    wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wndclass.lpfnWndProc = WndProc;
    wndclass.hInstance = hInstance;
    wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndclass.lpszClassName = TEXT("OpenGL Cloth Simulation");
    RegisterClassEx(&wndclass);

    // Create Window
    hwnd = CreateWindowEx(WS_EX_APPWINDOW, TEXT("OpenGL Cloth Simulation"), TEXT("Cloth Simulation"),
                          WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                          WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
    ghHWND = hwnd;

    initialize();
    ShowWindow(hwnd, iCmdShow);
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    // Main Loop
    while (!bDone)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                bDone = TRUE;
            }
            else
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
        else
        {
            if (gbActive)
            {
                display();
                update();
            }
        }
    }

    uninitialize();
    return (int)msg.wParam;
}

// Window Procedure
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    switch (iMsg)
    {
    case WM_SETFOCUS:
        gbActive = TRUE;
        break;
    case WM_KILLFOCUS:
        gbActive = FALSE;
        break;
    case WM_SIZE:
        glViewport(0, 0, LOWORD(lParam), HIWORD(lParam));
        break;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE)
            DestroyWindow(hwnd);
        break;
    case WM_CLOSE:
        DestroyWindow(hwnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hwnd, iMsg, wParam, lParam);
    }
    return 0;
}

// OpenGL Initialization
void initialize(void)
{
    PIXELFORMATDESCRIPTOR pfd;
    int iPixelFormatIndex;
    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 32;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;

    ghdc = GetDC(ghHWND);
    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
    SetPixelFormat(ghdc, iPixelFormatIndex, &pfd);
    ghrc = wglCreateContext(ghdc);
    wglMakeCurrent(ghdc, ghrc);
    glewInit();

    // Vertex Shader
    const GLchar *vertexShaderSource =
        "#version 410 core\n"
        "layout (location = 0) in vec4 position_mass;\n"
        "in vec2 aTexCoord;\n"
        "out vec2 oTexCoord;\n"
        "uniform float u_time;\n"
        "uniform float wave_amplitude;\n"
        "uniform float wave_frequency;\n"
        "uniform float wave_speed;\n"
        "void main() {\n"
        "    oTexCoord=aTexCoord;\n"
        "    float wave = wave_amplitude * sin(wave_frequency * position_mass.x + wave_speed * u_time);\n"
        "    gl_Position = position_mass + vec4(0.0, wave, 0.0, 0.0);\n"
        "}";

    // Fragment Shader
    const GLchar *fragmentShaderSource =
        "#version 410 core\n"
        "out vec4 FragColor;\n"
        "in vec2 oTexCoord;\n"
        "uniform sampler2D uTextureSampler;\n"
        "void main() {\n"
        "    FragColor=texture(uTextureSampler, oTexCoord);\n"
        "}";

    // Compile and Link Shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glBindAttribLocation(shaderProgramObject, AMC_ATTRIBUTE_TEXCOORD, "aTexCoord");

    shaderProgramObject = glCreateProgram();
    glAttachShader(shaderProgramObject, vertexShader);
    glAttachShader(shaderProgramObject, fragmentShader);
    glLinkProgram(shaderProgramObject);

    // Existing vertex buffer setup
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Generate cloth grid
    struct Vertex
    {
        vmath::vec4 position;
        vmath::vec2 texcoord;
    };

    // Allocate vertices & indices dynamically
    Vertex *vertices = new Vertex[POINTS_TOTAL];
    GLuint *indices = new GLuint[(POINTS_X - 1) * (POINTS_Y - 1) * 6];

    int index = 0; // ✅ Move outside of the loop
    for (int y = 0; y < POINTS_Y; y++)
    {
        for (int x = 0; x < POINTS_X; x++)
        {
            float fx = (float)x / (float)(POINTS_X - 1);
            float fy = (float)y / (float)(POINTS_Y - 1);

            float posX = fx * 2.0f - 1.0f;
            float posY = fy * 2.0f - 1.0f;

            // Store position and texcoords
            vertices[y * POINTS_X + x] = {vmath::vec4(posX, posY, 0.0f, 1.0f), vmath::vec2(fx, 1.0f - fy)};
        }
    }

    // ✅ Fix Double Declaration Issue
    index = 0;
    for (int y = 0; y < POINTS_Y - 1; y++)
    {
        for (int x = 0; x < POINTS_X - 1; x++)
        {
            int topLeft = y * POINTS_X + x;
            int topRight = y * POINTS_X + (x + 1);
            int bottomLeft = (y + 1) * POINTS_X + x;
            int bottomRight = (y + 1) * POINTS_X + (x + 1);

            indices[index++] = topLeft;
            indices[index++] = bottomLeft;
            indices[index++] = topRight;

            indices[index++] = bottomLeft;
            indices[index++] = bottomRight;
            indices[index++] = topRight;
        }
    }

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(false);
    unsigned char *data = stbi_load("cloth_texture.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
        glGenTextures(1, &Texture1);
        glBindTexture(GL_TEXTURE_2D, Texture1);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(data);

        printf("Texture loaded successfully: %d x %d\n", width, height);
    }
    else
    {
        printf("Failed to load texture\n");
    }

    // Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(Vertex), vertices, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    // Position Attribute
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);

    // Texcoord Attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, texcoord));
    glEnableVertexAttribArray(1);

    // Create and upload index buffer
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (POINTS_X - 1) * (POINTS_Y - 1) * 6 * sizeof(GLuint), indices, GL_STATIC_DRAW);

    delete[] vertices;
    delete[] indices;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
}

// Render Scene
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shaderProgramObject);

    glBindTexture(GL_TEXTURE_2D, Texture1);
    glUniform1i(textureSamplerUniform, 0);

    float currentTime = (float)(std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count());
    glUniform1f(glGetUniformLocation(shaderProgramObject, "u_time"), currentTime);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "wave_amplitude"), waveAmplitude);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "wave_frequency"), waveFrequency);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "wave_speed"), waveSpeed);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, (POINTS_X - 1) * (POINTS_Y - 1) * 6, GL_UNSIGNED_INT, 0);

    SwapBuffers(ghdc);
}

void update(void)
{
}

// Cleanup
void uninitialize(void)
{
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}
