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
        case 'W':
        case 'w':
            // Toggle wind
            {
                GLint windLoc = glGetUniformLocation(shaderProgramObject, "enable_wind");
                static bool windEnabled = false;
                windEnabled = !windEnabled;
                glProgramUniform1i(shaderProgramObject, windLoc, windEnabled);
                fprintf(gpFILE, "Wind %s\n", windEnabled ? "enabled" : "disabled");
            }
            break;
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
    // 1. Define the shader source
    const GLchar *fragmentShaderSourceCode = "#version 410 core\n"
                                             "in vec3 normal;\n"
                                             "in vec3 frag_position;\n"
                                             "out vec4 fragColor;\n"
                                             "uniform vec3 light_pos = vec3(10.0, 10.0, 10.0);\n"
                                             "uniform vec3 cloth_color = vec3(0.7, 0.2, 0.3);\n"
                                             "void main(void)\n"
                                             "{\n"
                                             "    vec3 norm = normalize(normal);\n"
                                             "    vec3 light_dir = normalize(light_pos - frag_position);\n"
                                             "    float diff = max(dot(norm, light_dir), 0.0);\n"
                                             "    vec3 diffuse = diff * cloth_color;\n"
                                             "    vec3 ambient = 0.1 * cloth_color;\n"
                                             "    fragColor = vec4(ambient + diffuse, 1.0);\n"
                                             "}\n";

    // 2. Create shader object
    GLuint fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

    // 3. Set source code
    glShaderSource(fragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

    // 4. Compile shader
    glCompileShader(fragmentShaderObject);

    // 5. Error checking
    GLint status = 0;
    GLint infoLogLength = 0;
    GLchar *szInfoLog = NULL;
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
                fprintf(gpFILE, "fragment shader compilation error log: %s\n", szInfoLog);
                free(szInfoLog);
                szInfoLog = NULL;
            }
        }
        uninitialize();
    }

    // 6-7. Attach to existing program or create new one
    // To attach to existing program:
    glAttachShader(shaderProgramObject, fragmentShaderObject);

    // 8. Link program (if adding to existing program)
    glLinkProgram(shaderProgramObject);

    // 9. Check linking errors
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

    // Allocate arrays for different spring types
    vmath::ivec4 *structural_springs = new vmath::ivec4[POINTS_TOTAL];
    vmath::ivec4 *shear_springs = new vmath::ivec4[POINTS_TOTAL];
    vmath::ivec4 *bend_springs = new vmath::ivec4[POINTS_TOTAL];

    // Initialize all to -1 (no connection)
    for (int i = 0; i < POINTS_TOTAL; i++)
    {
        structural_springs[i] = vmath::ivec4(-1);
        shear_springs[i] = vmath::ivec4(-1);
        bend_springs[i] = vmath::ivec4(-1);
    }

    // Set up spring connections
    for (int y = 0; y < POINTS_Y; y++)
    {
        for (int x = 0; x < POINTS_X; x++)
        {
            int idx = y * POINTS_X + x;

            // Structural springs (direct neighbors)
            int spring_count = 0;
            if (x > 0)
                structural_springs[idx][spring_count++] = idx - 1; // Left
            if (x < POINTS_X - 1)
                structural_springs[idx][spring_count++] = idx + 1; // Right
            if (y > 0)
                structural_springs[idx][spring_count++] = idx - POINTS_X; // Up
            if (y < POINTS_Y - 1)
                structural_springs[idx][spring_count++] = idx + POINTS_X; // Down

            // Shear springs (diagonal neighbors)
            spring_count = 0;
            if (x > 0 && y > 0)
                shear_springs[idx][spring_count++] = idx - POINTS_X - 1; // Upper-left
            if (x < POINTS_X - 1 && y > 0)
                shear_springs[idx][spring_count++] = idx - POINTS_X + 1; // Upper-right
            if (x > 0 && y < POINTS_Y - 1)
                shear_springs[idx][spring_count++] = idx + POINTS_X - 1; // Lower-left
            if (x < POINTS_X - 1 && y < POINTS_Y - 1)
                shear_springs[idx][spring_count++] = idx + POINTS_X + 1; // Lower-right

            // Bend springs (two particles away)
            spring_count = 0;
            if (x > 1)
                bend_springs[idx][spring_count++] = idx - 2; // Left 2
            if (x < POINTS_X - 2)
                bend_springs[idx][spring_count++] = idx + 2; // Right 2
            if (y > 1)
                bend_springs[idx][spring_count++] = idx - 2 * POINTS_X; // Up 2
            if (y < POINTS_Y - 2)
                bend_springs[idx][spring_count++] = idx + 2 * POINTS_X; // Down 2
        }
    }

    // Bind springs to shader attributes
    for (int i = 0; i < 2; i++)
    {
        glBindVertexArray(m_vao[i]);

        // Structural springs
        GLuint structuralVBO;
        glGenBuffers(1, &structuralVBO);
        glBindBuffer(GL_ARRAY_BUFFER, structuralVBO);
        glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(vmath::ivec4), structural_springs, GL_STATIC_DRAW);
        glVertexAttribIPointer(2, 4, GL_INT, 0, NULL);
        glEnableVertexAttribArray(2);

        // Shear springs
        GLuint shearVBO;
        glGenBuffers(1, &shearVBO);
        glBindBuffer(GL_ARRAY_BUFFER, shearVBO);
        glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(vmath::ivec4), shear_springs, GL_STATIC_DRAW);
        glVertexAttribIPointer(3, 4, GL_INT, 0, NULL);
        glEnableVertexAttribArray(3);

        // Bend springs
        GLuint bendVBO;
        glGenBuffers(1, &bendVBO);
        glBindBuffer(GL_ARRAY_BUFFER, bendVBO);
        glBufferData(GL_ARRAY_BUFFER, POINTS_TOTAL * sizeof(vmath::ivec4), bend_springs, GL_STATIC_DRAW);
        glVertexAttribIPointer(4, 4, GL_INT, 0, NULL);
        glEnableVertexAttribArray(4);
    }

    // Free memory
    delete[] structural_springs;
    delete[] shear_springs;
    delete[] bend_springs;

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

// Add to your display function for visualization
void renderCloth()
{
    // Use the positions from the current iteration
    glBindVertexArray(m_vao[m_iteration_index & 1]);

    // Draw the cloth as a mesh
    if (draw_lines)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, (POINTS_X - 1) * (POINTS_Y - 1) * 6, GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Draw control points if enabled
    if (draw_points)
    {
        glPointSize(3.0f);
        glDrawArrays(GL_POINTS, 0, POINTS_TOTAL);
    }
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

    // First pass: Physics simulation
    glUseProgram(shaderProgramObject);

    // Set uniforms
    glUniform1f(glGetUniformLocation(shaderProgramObject, "structural_k"), 500.0f);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "shear_k"), 50.0f);
    glUniform1f(glGetUniformLocation(shaderProgramObject, "bend_k"), 10.0f);
    // ... other uniforms ...

    // Enable transform feedback
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

    // Second pass: Rendering
    // (Switch to rendering shader program if needed)

    // Use the positions from the current iteration for rendering
    glBindVertexArray(m_vao[m_iteration_index & 1]);

    // Draw the cloth
    if (draw_lines)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, (POINTS_X - 1) * (POINTS_Y - 1) * 6, GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    else
    {
        glDrawElements(GL_TRIANGLES, (POINTS_X - 1) * (POINTS_Y - 1) * 6, GL_UNSIGNED_INT, 0);
    }

    // Draw control points if enabled
    if (draw_points)
    {
        glPointSize(3.0f);
        glDrawArrays(GL_POINTS, 0, POINTS_TOTAL);
    }

    glUseProgram(0);

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
