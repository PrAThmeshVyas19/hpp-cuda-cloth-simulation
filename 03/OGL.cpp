// common header files
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// OpenGL headers
#include <gl/glew.h>
#include <gl/Gl.h>

#include "OGL.h"
#include "vmath.h"

// CUDA headers
#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include\cuda_runtime.h>
#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include\cuda_gl_interop.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Image loading library

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")

// Cuda Related Lib
#pragma comment(lib, "cudart.lib")

using namespace vmath;

// Macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define POINTS_X 100
#define POINTS_Y 100
#define POINTS_TOTAL (POINTS_X * POINTS_Y)

// Global Variable declaration
FILE *gpFILE = NULL; // global pointer Type

HWND ghwnd = NULL; // global hwnd
BOOL gbActive = FALSE;
DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = {sizeof(WINDOWPLACEMENT)};
BOOL gbFullscreen = FALSE;

enum
{
    AMC_ATTRIBUTE_POSITION = 0
};

mat4 perspectiveProjectionMatrix; // mat4 is in vmath.h mat4 = 4*4 matrix
GLuint mvpMatrixUniform = 0;

cudaError_t cudaResult;
struct cudaGraphicsResource *cuda_graphics_resource = NULL;

// OpenGL Related Global Variables
HDC ghdc = NULL;
HGLRC ghrc = NULL;

GLuint textureID = 0;
GLuint shaderProgramObject = 0;

// Wave Parameters
float waveAmplitude = 0.1f;
float waveFrequency = 2.0f;
float waveSpeed = 3.0f;

GLuint vao = 0;
GLuint vbo = 0;

cudaGraphicsResource_t cudaVBO = nullptr;
GLuint vbo_position = 0;

// Time tracking
std::chrono::time_point<std::chrono::steady_clock> startTime;

// Function declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
int initialize(void);
void display(void);
void update(void);
void uninitialize(void);
void runCUDA(void);
void resize(int, int);

extern "C" void launchWaveSimulationKernel(float4 *pos, float time);

// Entry Point Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    // Function Declarations

    // Local Variable Declarations
    WNDCLASSEX wndclass;
    HWND hwnd;
    MSG msg;
    TCHAR szAppName[] = TEXT("PPVWindow");
    int iResult = 0;
    BOOL bDone = FALSE;

    // code
    if (fopen_s(&gpFILE, "log.txt", "w") != 0)
    {
        MessageBox(NULL, TEXT("Log file cannot be opened"), TEXT("Error"), MB_OK | MB_ICONERROR);
        exit(0);
    }

    fprintf(gpFILE, "Jay Ganeshay Namaha \nProgram Started Successfully \n");

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
                          TEXT("Prathmesh P. Vyas RTR5"),
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
            // case 'C':
            // case 'c':
            //     bOnGPU = FALSE;
            //     break;

            // case 'G':
            // case 'g':
            //     bOnGPU = TRUE;
            //     break;

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

// CUDA-OpenGL Interop Function
void runCUDA()
{
    cudaError_t error;
    float4 *devPos;
    size_t numBytes;

    // Map OpenGL VBO to CUDA
    error = cudaGraphicsMapResources(1, &cudaVBO);
    if (error != cudaSuccess)
    {
        fprintf(gpFILE, "cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(error));
        return;
    }

    error = cudaGraphicsResourceGetMappedPointer((void **)&devPos, &numBytes, cudaVBO);
    if (error != cudaSuccess)
    {
        fprintf(gpFILE, "cudaGraphicsResourceGetMappedPointer failed: %s\n", cudaGetErrorString(error));
        cudaGraphicsUnmapResources(1, &cudaVBO);
        return;
    }

    // Call the CUDA wrapper function
    launchWaveSimulationKernel(devPos, static_cast<float>(std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count()));

    // After kernel launch
    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(gpFILE, "CUDA kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    error = cudaGraphicsUnmapResources(1, &cudaVBO);
    if (error != cudaSuccess)
    {
        fprintf(gpFILE, "cudaGraphicsUnmapResources failed: %s\n", cudaGetErrorString(error));
    }
}

// Load Texture
void loadTexture()
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    int width, height, channels;
    unsigned char *image = stbi_load("cloth_texture.jpg", &width, &height, &channels, 4);
    if (image)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(image);
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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

// OpenGL Initialization
int initialize(void)
{
    // variable declarations
    PIXELFORMATDESCRIPTOR pfd;
    int iPixelFormatIndex = 0;

    // code
    // Check Cuda Support & if supported select default cuda device
    // int dev_count;
    // cudaResult = cudaGetDeviceCount(&dev_count);
    // if (cudaResult != cudaSuccess)
    // {
    //     fprintf(gpFILE, "cudaGetDeviceCount function failed !!!\n");
    //     uninitialize();
    //     exit(0);
    // }

    // else if (dev_count == 0)
    // {
    //     fprintf(gpFILE, "No Cuda Device!!!\n");
    //     uninitialize();
    //     exit(0);
    // }

    // // Success
    // else
    // {
    //     cudaSetDevice(0); // 0 is the default
    //     fprintf(gpFILE, "Cuda Success!!!\n");
    // }

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

    // // Step 4 Set optained pixel format
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
    glewInit();
    // if (glewInit() != GLEW_OK)
    // {
    //     fprintf(gpFILE, "glewInit() Failed !!!\n");
    //     return -6;
    // }

    // Print GLInfo
    printGLInfo();

    // loadTexture();

    // Shader Programs
    const GLchar *vertexShaderSourceCode =
        "#version 410 core\n"
        "layout (location = 0) in vec4 position;\n"
        "uniform mat4 uMVPMatrix;"
        "out vec2 TexCoord;\n"
        "void main() {\n"
        "    gl_Position = uMVPMatrix * position;\n"
        "    TexCoord = position.xy * 0.5 + 0.5;\n"
        "}";

    GLuint vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

    glCompileShader(vertexShaderObject);

    GLint status = 0;
    GLint infoLogLength = 0;
    GLchar *szInfoLog = NULL;

    glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(infoLogLength);
            if (szInfoLog != NULL)
            {
                glGetShaderInfoLog(vertexShaderObject, infoLogLength, NULL, szInfoLog);
                fprintf(gpFILE, "vertexShader Compilation Error Log : %s\n", szInfoLog);
                free(szInfoLog);
            }
        }
        uninitialize();
    }

    const GLchar *fragmentShaderSourceCode =
        "#version 410 core\n"
        "in vec2 TexCoord;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D clothTexture;\n"
        "void main() {\n"
        "    FragColor = texture(clothTexture, TexCoord);\n"
        "}";

    GLuint fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(fragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

    glCompileShader(fragmentShaderObject);

    status = 0;
    infoLogLength = 0;
    szInfoLog = NULL;

    glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(infoLogLength);
            if (szInfoLog != NULL)
            {
                glGetShaderInfoLog(fragmentShaderObject, infoLogLength, NULL, szInfoLog);
                fprintf(gpFILE, "FragmentShader Compilation Error Log : %s\n", szInfoLog);
                free(szInfoLog);
            }
        }
        uninitialize();
    }

    // Shader Program
    shaderProgramObject = glCreateProgram();

    glAttachShader(shaderProgramObject, vertexShaderObject);
    glAttachShader(shaderProgramObject, fragmentShaderObject);

    glBindAttribLocation(shaderProgramObject, AMC_ATTRIBUTE_POSITION, "position");

    glLinkProgram(shaderProgramObject);

    status = 0;
    infoLogLength = 0;
    szInfoLog = NULL;

    glGetProgramiv(shaderProgramObject, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramiv(shaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(infoLogLength);
            if (szInfoLog != NULL)
            {
                glGetProgramInfoLog(shaderProgramObject, infoLogLength, NULL, szInfoLog);
                fprintf(gpFILE, "Program Shader Compilation Error Log : %s\n", szInfoLog);
                free(szInfoLog);
                uninitialize();
            }
        }
    }

    // Get Uniform Shader Locations
    mvpMatrixUniform = glGetUniformLocation(shaderProgramObject, "uMVPMatrix");

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(float4), NULL, GL_DYNAMIC_DRAW);
    // cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Register OpenGL Buffer to Cuda Graphics Resource for InterOp
    // cudaResult = cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    // if (cudaResult != cudaSuccess)
    // {
    //     fprintf(gpFILE, "cudaGraphicsGLRegisterBuffer function failed !!!\n");
    //     uninitialize();
    //     exit(0);
    // }

    // Enabling Depth
    glClearDepth(1.0f);      // Depth buffer la clear karayla hii value vapar saglya bits la 1 kr
    glEnable(GL_DEPTH_TEST); // 8 test peiki enable kr depth test
    glDepthFunc(GL_LEQUAL);  // Depth sathi konta func vaparu test sathi LEQUAL = Less than or Equal to => ClearDepth 1.0f z coordinate

    // Set The ClearColor of Window to Blue
    // Here OpenGL Starts "Shree Gurudev Datta"
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Intialize Orthographic Projection Matrix
    perspectiveProjectionMatrix = vmath::mat4::identity();

    // Warmup Resize
    resize(WIN_WIDTH, WIN_HEIGHT);
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
    perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

// Render Cloth
void display(void)
{
    // Run CUDA operations first
    // runCUDA();

    // Then render with OpenGL
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shaderProgramObject);
    glUniformMatrix4fv(mvpMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix);

    glUniform1i(glGetUniformLocation(shaderProgramObject, "clothTexture"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glDrawArrays(GL_POINTS, 0, POINTS_TOTAL);
    glUseProgram(0);

    // Swap buffers last
    SwapBuffers(ghdc); // Make sure you're using ghdc, not ghwnd
}

void update(void)
{
    // Update simulation state if needed
}

// Cleanup
void uninitialize(void)
{
    // Delete shader program
    glDeleteProgram(shaderProgramObject);

    // Delete texture
    glDeleteTextures(1, &textureID);

    // // Unregister CUDA resources first
    // if (cudaVBO != nullptr)
    // {
    //     cudaGraphicsUnregisterResource(cudaVBO);
    //     cudaVBO = nullptr;
    // }

    // Delete OpenGL resources
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    // Release OpenGL context
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(ghrc);
    ReleaseDC(ghwnd, ghdc);
}
