cls

nvcc.exe -c -o cuda_cloth.cu.obj cuda_cloth.cu
 
cl.exe /c /EHsc /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" /I "C:\\glew-2.1.0\\include" OGL.cpp

rc.exe OGL.rc

link.exe OGL.obj cuda_cloth.cu.obj OGL.res user32.lib gdi32.lib OpenGL32.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" /LIBPATH:"C:\\glew\\lib\\Release\\x64" /SUBSYSTEM:WINDOWS