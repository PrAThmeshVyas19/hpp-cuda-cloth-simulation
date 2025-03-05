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

#include "OGL.h"

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
GLuint Texture2;

GLuint fadeAlphaUniform;
GLuint textureMixUniform;

const int NUM_TEXTURES = 4; // Change this based on how many textures you have
GLuint textures[NUM_TEXTURES];
int currentTextureIndex = 0;

BOOL gbFullscreen = FALSE;
DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = {sizeof(WINDOWPLACEMENT)};

static float fadeAlpha = 0.0f;
static float textureMix = 0.0f;
static bool fadeIn = true;

// Global Variable declaration
FILE *gpFILE = NULL; // global pointer Type

// Global variables
HWND ghwnd = NULL; // global hwnd
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
int initialize(void);
void display(void);
void update(void);
void uninitialize(void);
void printGLInfo(void);
void resize(int, int);

// Entry point function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    void update(void);

    // Local Variable Declarations
    WNDCLASSEX wndclass;
    HWND hwnd;
    MSG msg;
    TCHAR szAppName[] = TEXT("PPVWindow");
    int iResult = 0;
    BOOL bDone = FALSE;

    // code
    gpFILE = fopen("Log.txt", "w");
    if (gpFILE == NULL)
    {
        MessageBox(NULL, TEXT("Log file cannot be opened"), TEXT("Error"), MB_OK | MB_ICONERROR);
        exit(0);
    }

    fprintf(gpFILE, "Jay Ganeshay Namaha \n Program Started Successfully \n");

    // Wndclassex Initialization
    wndclass.cbSize = sizeof(WNDCLASSEX);
    wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wndclass.cbClsExtra = 0;
    wndclass.cbWndExtra = 0;
    wndclass.lpfnWndProc = WndProc;
    wndclass.hInstance = hInstance;
    wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
    wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndclass.lpszClassName = szAppName;
    wndclass.lpszMenuName = NULL;
    wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));

    // Register WNdclassex
    RegisterClassEx(&wndclass);

    // Create Window
    hwnd = CreateWindowEx(WS_EX_APPWINDOW,
                          szAppName,
                          TEXT("Cloth Simulation HPP"),
                          WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
                          CW_USEDEFAULT,
                          CW_USEDEFAULT,
                          WIN_WIDTH,
                          WIN_HEIGHT,
                          NULL,
                          NULL,
                          hInstance,
                          NULL);
    ghwnd = hwnd;

    // Initialization
    iResult = initialize();

    if (iResult != 0)
    {
        MessageBox(NULL, TEXT("Initialize() failed"), TEXT("Error"), MB_OK | MB_ICONERROR);
        DestroyWindow(hwnd);
    }

    // Show The WIndow
    ShowWindow(hwnd, iCmdShow);

    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    // Game Loop
    while (bDone == FALSE)
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
            if (gbActive == TRUE)
            {
                // Render
                display();

                // Update
                update();
            }
        }
    }

    // Uninitialization
    uninitialize();

    return ((int)msg.wParam);
}

// Call Back Function
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    // Function declarations
    void ToggleFullscreen(void);
    void resize(int, int);

    // code
    switch (iMsg)
    {
    case WM_SETFOCUS:
        gbActive = TRUE;
        break;

    case WM_KILLFOCUS:
        gbActive = FALSE;
        break;

    case WM_SIZE:
        resize(LOWORD(lParam), HIWORD(lParam)); // width ani height sangto eg. Awaj ala
        break;

    case WM_ERASEBKGND:
        return 0;

    case WM_KEYDOWN:
        switch (LOWORD(wParam))
        {
        case VK_ESCAPE:
            DestroyWindow(hwnd);
            break;
        }
        break;

    case WM_CHAR:
        switch (LOWORD(wParam))
        {
        case 'F':
        case 'f':
            if (gbFullscreen == FALSE)
            {
                ToggleFullscreen();
                gbFullscreen = TRUE;
            }
            else
            {
                ToggleFullscreen();
                gbFullscreen = FALSE;
            }
            break;
        }
        break;

    case WM_CLOSE:
        DestroyWindow(hwnd);
        break;

    case WM_DESTROY:
        if (gpFILE)
        {
            fprintf(gpFILE, "Program Ended Successfully \n");
        }
        PostQuitMessage(0);
        break;

    default:
        break;
    }

    return DefWindowProc(hwnd, iMsg, wParam, lParam);
}

void ToggleFullscreen(void)
{
    // local varaible declarations
    MONITORINFO mi = {sizeof(MONITORINFO)};

    // code
    if (gbFullscreen == FALSE)
    {
        dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

        if (dwStyle & WS_OVERLAPPEDWINDOW)
        {
            if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
            {
                SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
                SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED); // HWND_TOP == WS_OVERLAPPED , SWP_FRAMECHANGES => WM_NCCALSIZE(New Non client area calculate size)
            }
        }
        ShowCursor(FALSE);
    }

    else
    {
        SetWindowPlacement(ghwnd, &wpPrev);
        SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
        SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
        ShowCursor(TRUE);
    }
}
// OpenGL Initialization
int initialize(void)
{
    // variable declarations
    PIXELFORMATDESCRIPTOR pfd;
    int iPixelFormatIndex = 0;

    // code
    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

    // Initialization of pixel format descriptor
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cRedBits = 8;
    pfd.cGreenBits = 8;
    pfd.cBlueBits = 8;
    pfd.cAlphaBits = 8;
    pfd.cDepthBits = 32;

    // Step 2 Get the device context painter
    ghdc = GetDC(ghwnd);
    if (ghdc == NULL)
    {
        fprintf(gpFILE, "GetDC() Failed !!!\n");
        return -1;
    }

    // Step 3
    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
    if (iPixelFormatIndex == 0)
    {
        fprintf(gpFILE, "ChoosePixelFormat() Failed !!!\n");
        return -2;
    }

    // Step 4 Set optained pixel format
    if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
    {
        fprintf(gpFILE, "SetPixelFormat() Failed!!!\n");
        return -3;
    }

    // Step 5 Tell WindowGraphicLibrary to give me OpenGL compatible dc from this dc
    ghrc = wglCreateContext(ghdc);
    if (ghrc == NULL)
    {
        fprintf(gpFILE, "wglCreateContext() Failed !!!\n");
        return -4;
    } // now gldc will end its role and give controll to ghrc

    // Make Rendering Context Current
    if (wglMakeCurrent(ghdc, ghrc) == FALSE)
    {
        fprintf(gpFILE, "wglMakeCurrent() Failed !!!\n");
        return -5;
    }

    // intialize GLEW
    if (glewInit() != GLEW_OK)
    {
        fprintf(gpFILE, "glewInit() Failed !!!\n");
        return -6;
    }

    // Print GLInfo
    printGLInfo();

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
        "    oTexCoord = aTexCoord;\n"
        "    \n"
        "    // Compute distance from center (0,0) in XZ plane\n"
        "    float distance = length(position_mass.xz);\n"
        "    \n"
        "float decay = exp(-0.1 * distance);"
        "    // Create ripple effect using sine wave\n"
        "    float wave = decay * wave_amplitude * sin(wave_frequency * distance - wave_speed * u_time);\n"
        "    \n"
        "    // Move the vertex up and down to simulate ripples\n"
        "    gl_Position = position_mass + vec4(0.0, wave, 0.0, 0.0);\n"
        "}";

    // Fragment Shader
    const GLchar *fragmentShaderSource =
        "#version 410 core\n"
        "out vec4 FragColor;\n"
        "in vec2 oTexCoord;\n"
        "uniform sampler2D uTexture;\n"
        "uniform float fadeAlpha;\n"
        "void main() {\n"
        "    vec4 texColor = texture(uTexture, oTexCoord);\n"
        "    FragColor = vec4(texColor.rgb, fadeAlpha);  // Apply fadeAlpha to RGB and Alpha\n"
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

    const char *textureFiles[NUM_TEXTURES] = {
        "cloth_texture2.jpg",
        "cloth_texture3.jpg",
        "cloth_texture4.jpg",
        "cloth_texture.jpg"};

    for (int i = 0; i < NUM_TEXTURES; i++)
    {
        int width, height, nrChannels;
        unsigned char *data = stbi_load(textureFiles[i], &width, &height, &nrChannels, 0);
        if (data)
        {
            glGenTextures(1, &textures[i]);
            glBindTexture(GL_TEXTURE_2D, textures[i]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            stbi_image_free(data);
        }
    }

    // Get uniform locations
    fadeAlphaUniform = glGetUniformLocation(shaderProgramObject, "fadeAlpha");
    textureMixUniform = glGetUniformLocation(shaderProgramObject, "textureMix");

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

    // Set The ClearColor of Window to Blue
    // Here OpenGL Starts "Shree Gurudev Datta"
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Warmup Resize
    resize(WIN_WIDTH, WIN_HEIGHT);

    return 0;
}

void printGLInfo(void)
{
    // Variable Declarations
    GLint numExtentions;
    GLint i;

    // code
    fprintf(gpFILE, "\n***********************************************************************************\n");

    fprintf(gpFILE, "OpenGL Vender : %s\n", glGetString(GL_VENDOR));

    fprintf(gpFILE, "OpenGL Renderer : %s\n", glGetString(GL_RENDERER));

    fprintf(gpFILE, "OpenGL Version : %s\n", glGetString(GL_VERSION));

    fprintf(gpFILE, "GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION)); // Graphic Library Shading Language

    // Listing of Supported Extentions
    fprintf(gpFILE, "\n***********************************************************************************\n");
    glGetIntegerv(GL_NUM_EXTENSIONS, &numExtentions);

    for (i = 0; i < numExtentions; i++)
    {
        fprintf(gpFILE, "%s\n", glGetStringi(GL_EXTENSIONS, i));
    }

    fprintf(gpFILE, "\n***********************************************************************************\n");
}

void resize(int width, int height)
{
    // code
    if (height <= 0)
    {
        height = 1;
    }

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    // Set Perspective projection matrix
    // perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

// Render Scene
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shaderProgramObject);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind the current texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textures[currentTextureIndex]);
    glUniform1i(glGetUniformLocation(shaderProgramObject, "uTexture"), 0);

    // Send fadeAlpha to the shader
    glUniform1f(fadeAlphaUniform, fadeAlpha);

    float currentTime = (float)(std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count());
    glUniform1f(glGetUniformLocation(shaderProgramObject, "u_time"), currentTime);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "wave_amplitude"), waveAmplitude);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "wave_frequency"), waveFrequency);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "wave_speed"), waveSpeed);

    glUniform1f(fadeAlphaUniform, fadeAlpha);
    glUniform1f(textureMixUniform, textureMix);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, (POINTS_X - 1) * (POINTS_Y - 1) * 6, GL_UNSIGNED_INT, 0);

    SwapBuffers(ghdc);
}

void update(void)
{
    float fadeSpeed = 0.001f; // Adjust speed for smooth fade effect

    if (fadeIn)
    {
        fadeAlpha += fadeSpeed;
        if (fadeAlpha >= 1.0f)
        {
            fadeAlpha = 1.0f;
            fadeIn = false; // Start fading out
        }
    }
    else
    {
        fadeAlpha -= fadeSpeed;
        if (fadeAlpha <= 0.0f)
        {
            fadeAlpha = 0.0f;
            fadeIn = true; // Start fading in the next texture

            // Move to the next texture
            currentTextureIndex = (currentTextureIndex + 1) % NUM_TEXTURES;
        }
    }
}

// Cleanup
void uninitialize(void)
{
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}
