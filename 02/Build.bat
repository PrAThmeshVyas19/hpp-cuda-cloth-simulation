cls


cl.exe /c /EHsc /I "C:\\glew\\include" OGL.cpp


rc.exe OGL.rc


link.exe OGL.obj OGL.res user32.lib gdi32.lib OpenGL32.lib /LIBPATH:"C:\\glew\\lib\\Release\\x64" /SUBSYSTEM:WINDOWS