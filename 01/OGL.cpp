// common header files
#include <windows.h> // win32 API
#include <stdio.h>   // for file I/O
#include <stdlib.h>  // for exit()
// #define _USE_MATH_DEFINES
// #include <math.h> // for M_PI
#include <chrono>

// OpenGL header files
#include <gl/glew.h> // This must be before gl/Gl.h
#include <gl/Gl.h>

#include "vmath.h"
using namespace vmath;

#include "OGL.h"

// Link with OpenGL
#pragma comment(lib, "opengl32.lib") // open graphics library
#pragma comment(lib, "glew32.lib")

// Macros
#define WIN_WIDTH 800
#define WIN_HEIGHT 600

// global function declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// global variable declarations
FILE *gpFILE = NULL;
std::chrono::time_point<std::chrono::steady_clock> startTime;

HWND ghHWND = NULL;
DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = {sizeof(WINDOWPLACEMENT)};
BOOL gbActive = FALSE;
BOOL gbFullscreen = FALSE;

// OpenGL related global variables
HDC ghdc = NULL;
HGLRC ghrc = NULL;

enum
{
    AMC_ATTRIBUTE_POSITION = 0,
    AMC_ATTRIBUTE_COLOR,
    AMC_ATTRIBUTE_TEXTURE_COORDINATES,
    AMC_ATTRIBUTE_NORMAL,
    AMC_ATTRIBUTE_OFFSET
};
GLuint shaderProgramObject = 0;
GLuint shaderProgramObject_controlPoints = 0;
mat4 perspectiveProjectionMatrix; // mat4 is in vmath.h

enum BUFFER_TYPE_t
{
    POSITION_A,
    POSITION_B,
    VELOCITY_A,
    VELOCITY_B,
    CONNECTION
};
enum
{
    POINTS_X = 50,
    POINTS_Y = 50,
    POINTS_TOTAL = (POINTS_X * POINTS_Y),
    CONNECTIONS_TOTAL = (POINTS_X - 1) * POINTS_Y + (POINTS_Y - 1) * POINTS_X
};
GLuint m_vao[2];
GLuint m_vbo[5];
GLuint m_index_buffer;
GLuint m_pos_tbo[2];
GLuint m_C_loc;
GLuint m_iteration_index = 0;

bool draw_points = true;
bool draw_lines = true;
int iterations_per_frame = 16;

// entry point function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    // function declarations
    int initialize(void);
    void uninitialize(void);
    void display(void);
    void update(void);

    // local variable declarations
    WNDCLASSEX wndclass;
    HWND hwnd;
    MSG msg;
    TCHAR szAppName[] = TEXT("VNDWindow");
    int iResult = 0;
    BOOL bDone = FALSE;

    // code
    if (fopen_s(&gpFILE, "log.txt", "w") != 0)
    {
        MessageBox(NULL, TEXT("log file can not be opened"), TEXT("error"), MB_OK | MB_ICONERROR);
        exit(0);
    }
    fprintf(gpFILE, "Program started successfully!\n");
    fprintf(gpFILE, "**************************************************************************************\n");

    // WNDCLASSEX initialization
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

    // register WNDCLASSEX
    RegisterClassEx(&wndclass);

    // Get the screen dimensions
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    // Calculate the window size and position
    int x = (screenWidth - WIN_WIDTH) / 2;
    int y = (screenHeight - WIN_HEIGHT) / 2;

    // create the window
    hwnd = CreateWindowEx(
        WS_EX_APPWINDOW,
        szAppName,
        TEXT("Vaishali Dudhmal"),
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
        x,
        y,
        WIN_WIDTH,
        WIN_HEIGHT,
        NULL,
        NULL,
        hInstance,
        NULL);
    ghHWND = hwnd;

    // initialization
    iResult = initialize();
    if (iResult != 0)
    {
        MessageBox(hwnd, TEXT("initialize() failed"), TEXT("error"), MB_OK | MB_ICONERROR);
        DestroyWindow(hwnd);
        exit(0);
    }

    // show the window
    ShowWindow(hwnd, iCmdShow);
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    // game loop
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

                // update
                update();
            }
        }
    }

    // uninitialization
    uninitialize();

    return ((int)msg.wParam);
}

// callback function
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    // function declarations
    void toggleFullscreen(void);
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
        resize(LOWORD(lParam), HIWORD(lParam));
        break;
    case WM_ERASEBKGND:
        return (0);
    case WM_KEYDOWN:
        switch (LOWORD(wParam))
        {
        case VK_ESCAPE:
            DestroyWindow(hwnd);
            break;
        case VK_UP:
            iterations_per_frame++;
            break;
        case VK_DOWN:
            iterations_per_frame--;
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
                toggleFullscreen();
                gbFullscreen = TRUE;
            }
            else
            {
                toggleFullscreen();
                gbFullscreen = FALSE;
            }
            break;
        case 'P':
        case 'p':
            draw_points = !draw_points;
            break;
        case 'L':
        case 'l':
            draw_lines = !draw_lines;
            break;
        }
        break;
    case WM_CLOSE:
        DestroyWindow(hwnd);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        break;
    }

    return (DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void toggleFullscreen(void)
{
    // local variable declarations
    MONITORINFO mi = {sizeof(MONITORINFO)};

    // code
    if (gbFullscreen == FALSE)
    {
        dwStyle = GetWindowLong(ghHWND, GWL_STYLE);
        if (dwStyle & WS_OVERLAPPEDWINDOW)
        {
            if (GetWindowPlacement(ghHWND, &wpPrev) &&
                GetMonitorInfo(MonitorFromWindow(ghHWND, MONITORINFOF_PRIMARY), &mi))
            {
                SetWindowLong(ghHWND, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
                SetWindowPos(ghHWND, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
            }
        }
        ShowCursor(FALSE);
    }
    else
    {
        SetWindowPlacement(ghHWND, &wpPrev);
        SetWindowLong(ghHWND, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
        SetWindowPos(ghHWND, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
        ShowCursor(TRUE);
    }
}

int initialize(void)
{
    // function declarations
    void printGLInfo();
    void uninitialize();
    void resize(int, int);
    BOOL loadGLTexture(GLuint * texture, TCHAR imageResourceId[]);

    // code
    PIXELFORMATDESCRIPTOR pfd;
    int iPixelFormatIndex = 0;
    BOOL bResult = FALSE;

    // initialize pfd
    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
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

    // Get DC
    ghdc = GetDC(ghHWND);
    if (ghdc == NULL)
    {
        fprintf(gpFILE, "GetDC() failed!\n");
        return -1;
    }

    // Get closes pixelfomat matching the one provided
    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
    if (iPixelFormatIndex == 0)
    {
        fprintf(gpFILE, "ChoosePixelFormat() failed!\n");
        return -2;
    }

    if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
    {
        fprintf(gpFILE, "SetPixelFomat() failed!\n");
        return -3;
    }

    // Tell WGL(bridging libary) to give me OpenGL compatible DC
    ghrc = wglCreateContext(ghdc);
    if (ghrc == NULL)
    {
        fprintf(gpFILE, "wglCreateContext() failed!\n");
        return -4;
    }

    // make rendering context current, ghdc -> ghrc control handover
    if (wglMakeCurrent(ghdc, ghrc) == FALSE)
    {
        fprintf(gpFILE, "wglMakeCurrent() failed!\n");
        return -5;
    }

    // initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        fprintf(gpFILE, "glewInit() failed!\n");
        return -6;
    }

    printGLInfo();

    // vertex shader
    const GLchar *vertexShaderSourceCode =
        "#version 410 core"
        "\n"
        "layout (location = 0) in vec4 position_mass;"
        "layout (location = 1) in vec3 velocity;"
        "layout (location = 2) in ivec4 connection;"
        "uniform samplerBuffer tex_position;"
        "out vec4 tf_position_mass;"
        "out vec3 tf_velocity;"
        "uniform float t = 0.07;"
        "uniform float k = 7.1;"
        "const vec3 gravity = vec3(0.0, -0.08, 0.0);"
        "uniform float c = 2.8;"
        "uniform float rest_length = 0.88;"
        "void main(void)"
        "{"
        "vec3 p = position_mass.xyz;"
        "float m = position_mass.w;"
        "vec3 u = velocity;"
        "vec3 F = gravity * m - c * u;"
        "bool fixed_node = true;"
        "for (int i = 0; i < 4; i++) {"
        "if (connection[i] != -1) {"
        "vec3 q = texelFetch(tex_position, connection[i]).xyz;"
        "vec3 d = q - p;"
        "float x = length(d);"
        "F += -k * (rest_length - x) * normalize(d);"
        "fixed_node = false;"
        "}"
        "}"
        "if (fixed_node)"
        "{"
        "F = vec3(0.0);"
        "}"
        "vec3 a = F / m;"
        "vec3 s = u * t + 0.5 * a * t * t;"
        "vec3 v = u + a * t;"
        "s = clamp(s, vec3(-25.0), vec3(25.0));"
        "tf_position_mass = vec4(p + s, m);"
        "tf_velocity = v;"
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
                fprintf(gpFILE, "vertex shader compilation error log: %s\n", szInfoLog);
                free(szInfoLog);
                szInfoLog = NULL;
            }
        }
        uninitialize();
    }

    // Shader program
    shaderProgramObject = glCreateProgram();
    glAttachShader(shaderProgramObject, vertexShaderObject);
    // glAttachShader(shaderProgramObject, fragmentShaderObject);
    static const char *tf_varyings[] = {"tf_position_mass", "tf_velocity"};
    glTransformFeedbackVaryings(shaderProgramObject, 2, tf_varyings, GL_SEPARATE_ATTRIBS);
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
                fprintf(gpFILE, "shader program linking error log: %s\n", szInfoLog);
                free(szInfoLog);
                szInfoLog = NULL;
            }
        }
        uninitialize();
    }

    // vertex shader
    const GLchar *vertexShaderSourceCode_controlPoints =
        "#version 410 core"
        "\n"
        "in vec3 aPosition;"
        "void main(void)"
        "{"
        "gl_Position = vec4(aPosition * 0.03, 1.0);"
        "}";
    GLuint vertexShaderObject_controlPoints = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderObject_controlPoints, 1, (const GLchar **)&vertexShaderSourceCode_controlPoints, NULL);
    glCompileShader(vertexShaderObject_controlPoints);
    status = 0;
    infoLogLength = 0;
    szInfoLog = NULL;
    glGetShaderiv(vertexShaderObject_controlPoints, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObject_controlPoints, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(infoLogLength);
            if (szInfoLog != NULL)
            {
                glGetShaderInfoLog(vertexShaderObject_controlPoints, infoLogLength, NULL, szInfoLog);
                fprintf(gpFILE, "vertex shader compilation error log: %s\n", szInfoLog);
                free(szInfoLog);
                szInfoLog = NULL;
            }
        }
        uninitialize();
    }

    // fragment shader
    const GLchar *fragmentShaderSourceCode_controlPoints =
        "#version 410 core"
        "\n"
        "out vec4 fragColor;"
        "void main(void)"
        "{"
        "fragColor = vec4(1.0 , 0.0f , 0.0f , 1.0f );"
        "}";
    GLuint fragmentShaderObject_controlPoints = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderObject_controlPoints, 1, (const GLchar **)&fragmentShaderSourceCode_controlPoints, NULL);
    glCompileShader(fragmentShaderObject_controlPoints);
    status = 0;
    infoLogLength = 0;
    szInfoLog = NULL;
    glGetShaderiv(fragmentShaderObject_controlPoints, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObject_controlPoints, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(infoLogLength);
            if (szInfoLog != NULL)
            {
                glGetShaderInfoLog(fragmentShaderObject_controlPoints, infoLogLength, NULL, szInfoLog);
                fprintf(gpFILE, "fragment shader compilation error log: %s\n", szInfoLog);
                free(szInfoLog);
                szInfoLog = NULL;
            }
        }
        uninitialize();
    }

    // Shader program
    shaderProgramObject_controlPoints = glCreateProgram();
    glAttachShader(shaderProgramObject_controlPoints, vertexShaderObject_controlPoints);
    glAttachShader(shaderProgramObject_controlPoints, fragmentShaderObject_controlPoints);
    glBindAttribLocation(shaderProgramObject, AMC_ATTRIBUTE_POSITION, "aPosition");
    glLinkProgram(shaderProgramObject_controlPoints);
    status = 0;
    infoLogLength = 0;
    szInfoLog = NULL;
    glGetProgramiv(shaderProgramObject_controlPoints, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramiv(shaderProgramObject_controlPoints, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szInfoLog = (GLchar *)malloc(infoLogLength);
            if (szInfoLog != NULL)
            {
                glGetProgramInfoLog(shaderProgramObject_controlPoints, infoLogLength, NULL, szInfoLog);
                fprintf(gpFILE, "shader program linking error log: %s\n", szInfoLog);
                free(szInfoLog);
                szInfoLog = NULL;
            }
        }
        uninitialize();
    }

    vmath::vec4 *initial_positions = new vmath::vec4[POINTS_TOTAL];
    vmath::vec3 *initial_velocities = new vmath::vec3[POINTS_TOTAL];
    vmath::ivec4 *connection_vectors = new vmath::ivec4[POINTS_TOTAL];

    int n = 0;
    for (int j = 0; j < POINTS_Y; j++)
    {
        float fj = (float)j / (float)POINTS_Y;
        for (int i = 0; i < POINTS_X; i++)
        {
            float fi = (float)i / (float)POINTS_X;

            initial_positions[n] = vmath::vec4((fi - 0.5f) * (float)POINTS_X, (fj - 0.5f) * (float)POINTS_Y, 0.6f * sinf(fi) * cosf(fj), 1.0f);
            initial_velocities[n] = vmath::vec3(0.0f);
            connection_vectors[n] = vmath::ivec4(-1);

            if (j != (POINTS_Y - 1))
            {
                if (i != 0)
                    connection_vectors[n][0] = n - 1;

                if (j != 0)
                    connection_vectors[n][1] = n - POINTS_X;

                if (i != (POINTS_X - 1))
                    connection_vectors[n][2] = n + 1;

                if (j != (POINTS_Y - 1))
                    connection_vectors[n][3] = n + POINTS_X;
            }
            n++;
        }
    }

    glGenVertexArrays(2, m_vao);
    glGenBuffers(5, m_vbo);

    for (int i = 0; i < 2; i++)
    {
        glBindVertexArray(m_vao[i]);

        glBindBuffer(GL_ARRAY_BUFFER, m_vbo[POSITION_A + i]);
        glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(vmath::vec4), initial_positions, GL_DYNAMIC_COPY);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, m_vbo[VELOCITY_A + i]);
        glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(vmath::vec3), initial_velocities, GL_DYNAMIC_COPY);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, m_vbo[CONNECTION]);
        glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(vmath::ivec4), connection_vectors, GL_STATIC_DRAW);
        glVertexAttribIPointer(2, 4, GL_INT, 0, NULL);
        glEnableVertexAttribArray(2);
    }

    delete[] connection_vectors;
    delete[] initial_velocities;
    delete[] initial_positions;

    glGenTextures(2, m_pos_tbo);
    glBindTexture(GL_TEXTURE_BUFFER, m_pos_tbo[0]);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, m_vbo[POSITION_A]);
    glBindTexture(GL_TEXTURE_BUFFER, m_pos_tbo[1]);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, m_vbo[POSITION_B]);

    int lines = (POINTS_X - 1) * POINTS_Y + (POINTS_Y - 1) * POINTS_X;

    glGenBuffers(1, &m_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, lines * 2 * sizeof(int), NULL, GL_STATIC_DRAW);
    int *e = (int *)glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, lines * 2 * sizeof(int), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    for (int j = 0; j < POINTS_Y; j++)
    {
        for (int i = 0; i < POINTS_X - 1; i++)
        {
            *e++ = i + j * POINTS_X;
            *e++ = 1 + i + j * POINTS_X;
        }
    }
    for (int i = 0; i < POINTS_X; i++)
    {
        for (int j = 0; j < POINTS_Y - 1; j++)
        {
            *e++ = i + j * POINTS_X;
            *e++ = POINTS_X + i + j * POINTS_X;
        }
    }
    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);

    startTime = std::chrono::steady_clock::now();

    // Enable depth
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Set the clearcolor of window to black
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // initialize perspectiveProjectionMatrix
    perspectiveProjectionMatrix = vmath::mat4::identity();

    // warmup
    resize(WIN_WIDTH, WIN_HEIGHT);

    return (0);
}

void printGLInfo()
{
    // variable declarations
    GLint numExtensions;
    GLint i;

    // code
    fprintf(gpFILE, "OpenGL vendor: %s\n", glGetString(GL_VENDOR));
    fprintf(gpFILE, "OpenGL renderer: %s\n", glGetString(GL_RENDERER));
    fprintf(gpFILE, "OpenGL version: %s\n", glGetString(GL_VERSION));
    fprintf(gpFILE, "GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    fprintf(gpFILE, "**************************************************************************************\n");

    // listing of supported extensions
    glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
    for (int i = 0; i < numExtensions; i++)
    {
        fprintf(gpFILE, "%s\n", glGetStringi(GL_EXTENSIONS, i));
    }
    fprintf(gpFILE, "**************************************************************************************\n");
}

void resize(int width, int height)
{
    // code
    if (height <= 0)
    {
        height = 1;
    }

    if (width <= 0)
    {
        width = 1;
    }

    // Viewport == binocular
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    // set perspective projection matrix
    perspectiveProjectionMatrix = vmath::perspective(45.0f, ((GLfloat)width / (GLfloat)height), 0.1f, 100.0f);
}

double getCurrentTime()
{
    auto endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedTime{endTime - startTime};
    return elapsedTime.count();
}

void display(void)
{
    // code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgramObject);

    glEnable(GL_RASTERIZER_DISCARD);

    for (int i = iterations_per_frame; i != 0; --i)
    {
        glBindVertexArray(m_vao[m_iteration_index & 1]);
        glBindTexture(GL_TEXTURE_BUFFER, m_pos_tbo[m_iteration_index & 1]);
        m_iteration_index++;
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_vbo[POSITION_A + (m_iteration_index & 1)]);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, m_vbo[VELOCITY_A + (m_iteration_index & 1)]);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, POINTS_TOTAL);
        glEndTransformFeedback();
    }

    glDisable(GL_RASTERIZER_DISCARD);

    // show control points and cage
    {
        glUseProgram(shaderProgramObject_controlPoints);

        // if (draw_points)
        // {
        //     glPointSize(4.0f);
        //     glDrawArrays(GL_POLYGON, 0, POINTS_TOTAL);
        // }

        if (draw_lines)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_buffer);
            glDrawElements(GL_POLYGON, CONNECTIONS_TOTAL * 2, GL_UNSIGNED_INT, NULL);
        }

        glUseProgram(0);
    }

    SwapBuffers(ghdc);
}

void update(void)
{
    // code
}

void uninitialize(void)
{
    // function declarations
    void toggleFullscreen(void);

    // code
    if (shaderProgramObject)
    {
        glUseProgram(shaderProgramObject);
        GLint numShaders = 0;
        glGetProgramiv(shaderProgramObject, GL_ATTACHED_SHADERS, &numShaders);
        if (numShaders > 0)
        {
            GLuint *pShaders = (GLuint *)malloc(numShaders * sizeof(GLuint));
            if (pShaders != NULL)
            {
                glGetAttachedShaders(shaderProgramObject, numShaders, NULL, pShaders);
                for (GLint i = 0; i < numShaders; i++)
                {
                    glDetachShader(shaderProgramObject, pShaders[i]);
                    glDeleteShader(pShaders[i]);
                    pShaders[i] = 0;
                }
                free(pShaders);
                pShaders = NULL;
            }
        }
        glUseProgram(0);
        glDeleteProgram(shaderProgramObject);
        shaderProgramObject = 0;
    }
    glDeleteBuffers(5, m_vbo);
    glDeleteVertexArrays(2, m_vao);

    // when application existing in fullscrreen
    if (gbFullscreen == TRUE)
    {
        toggleFullscreen();
        gbFullscreen = FALSE;
    }

    // Make HDC cuurent
    if (wglGetCurrentContext() == ghrc)
    {
        wglMakeCurrent(NULL, NULL);
    }

    // delete redering context
    if (ghrc)
    {
        wglDeleteContext(ghrc);
    }

    // Release hdc
    if (ghdc)
    {
        ReleaseDC(ghHWND, ghdc);
    }

    // destroy window
    if (ghHWND)
    {
        DestroyWindow(ghHWND);
        ghHWND = NULL;
    }

    // close file
    if (gpFILE)
    {
        fprintf(gpFILE, "Program ended successfully!\n");
        fprintf(gpFILE, "**************************************************************************************\n");
        fclose(gpFILE);
        gpFILE = NULL;
    }
}
