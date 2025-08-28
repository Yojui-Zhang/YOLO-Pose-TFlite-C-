#ifdef _opengl
	#include <iostream>
	#include <string.h>
	#include <gc_vdk.h>
	#include <GLES2/gl2.h>
	#include <GLES2/gl2ext.h>
	#include <GLES3/gl3.h>
	#include <GLES3/gl3ext.h>

	using namespace std;

	#define TUTORIAL_NAME "OpenGL ES 2.0 Tutorial 1"
	#define EGL_WIDTH 1920
	#define EGL_HEIGHT 1080
	vdkEGL egl;

	void init_egl(void)
	{
		printf("v1\n");
		EGLint EGLerror = 0;
		EGLint configAttribs[] =
			{
				EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
				EGL_SAMPLES,      0,
				EGL_BUFFER_SIZE, 24,
				EGL_DEPTH_SIZE, 1,
				EGL_RED_SIZE, 8,
				EGL_GREEN_SIZE, 8,
				EGL_BLUE_SIZE, 8,
				EGL_ALPHA_SIZE, 8,
				EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
				EGL_NONE
			};

		EGLint attribListContext[] =
		{
			// Needs to be set for es2.0 as default client version is es1.1.
			EGL_CONTEXT_CLIENT_VERSION, 2,
			EGL_NONE
		};

		if (!vdkSetupEGL(0, 0, EGL_WIDTH, EGL_HEIGHT, configAttribs, NULL, attribListContext, &egl)) //1920 1080
		{
			printf("EGL Error");
			// return 1;
		}

		vdkSetWindowTitle(egl.window, TUTORIAL_NAME);
		vdkShowWindow(egl.window);
	}
	void swap_egl(void)
	{
		vdkSwapEGL(&egl);
	}
	void finish_egl(void)
	{
		vdkFinishEGL(&egl);
	}

	class glProgramObject
	{
	public:
		void loadShaders(string vShaderFName, string pShaderFName)
		{
			GLint linked = 0;
			int ret = 0;
			char * infoLog;

			vertShader = glCreateShader(GL_VERTEX_SHADER);
			fragShader = glCreateShader(GL_FRAGMENT_SHADER);
			program = glCreateProgram();

			CompileShader(vShaderFName.data(), vertShader);
			CompileShader(pShaderFName.data(), fragShader);

			glAttachShader(program, vertShader);
			glAttachShader(program, fragShader);
			glLinkProgram(program);

			glGetProgramiv(program, GL_LINK_STATUS, &linked);
			if (!linked)
			{
				GLint errorBufSize, errorLength;
				glGetShaderiv(program, GL_INFO_LOG_LENGTH, &errorBufSize);
				infoLog = new char[errorBufSize+1];
				if (!infoLog)
				{
				// Retrieve error.
					glGetProgramInfoLog(program, errorBufSize, &errorLength, infoLog);
					infoLog[errorBufSize + 1] = '\0';
					fprintf(stderr, "%s", infoLog);
					delete [] infoLog;
				}
				fprintf(stderr, "Error linking program\n");
				return;
			}
			glUseProgram(program);
		}
		GLuint getProgram()
		{
			return GLuint(program);
		}
		void DestroyShaders()
		{
			glDeleteShader(vertShader);
			glDeleteShader(fragShader);
			glDeleteProgram(program);
			glUseProgram(0);
		}
	private:
		int CompileShader(const char * FName, GLuint ShaderNum)
		{
			FILE * fptr = NULL;
			GLint compiled = 0;
			int ret = 0;
			int length;
			char * shaderSource;
			char * infoLog;

			fptr = fopen(FName, "rb");
			if (fptr == NULL)
			{
				fprintf(stderr, "Cannot open file '%s'\n", FName);
				return 0;
			}
			fseek(fptr, 0, SEEK_END);
			length = ftell(fptr);
			fseek(fptr, 0 ,SEEK_SET);

			shaderSource = new char[length];
			if (shaderSource == NULL)
			{
				fprintf(stderr, "Out of memory.\n");
				return 0;
			}

			ret = fread(shaderSource, length, 1, fptr);

			glShaderSource(ShaderNum, 1, (const char**)&shaderSource, &length);
			glCompileShader(ShaderNum);
			
			delete [] shaderSource;
			fclose(fptr);

			glGetShaderiv(ShaderNum, GL_COMPILE_STATUS, &compiled);
			if (!compiled)
			{
				// Retrieve error buffer size.
				GLint errorBufSize, errorLength;
				glGetShaderiv(ShaderNum, GL_INFO_LOG_LENGTH, &errorBufSize);
				infoLog = new char[errorBufSize+1];
				if (!infoLog)
				{
					// Retrieve error.
					glGetShaderInfoLog(ShaderNum, errorBufSize, &errorLength, infoLog);
					infoLog[errorBufSize + 1] = '\0';
					fprintf(stderr, "%s\n", infoLog);

					delete [] infoLog;
				}
				fprintf(stderr, "Error compiling shader '%s'\n", FName);
				return 0;
			}

			return 1;
		}
		GLuint vertShader;
		GLuint fragShader;
		GLuint program;
	}progImageShow;

	void glinit(void)
	{
		init_egl();
		progImageShow.loadShaders("imageShow.vert", "imageShow.frag");
	}

	int screenWidth = 1920;
	int screenHeight = 1080;
	static int flag = 0;

	void imageShow(int width, int height, unsigned char rgb[])
	{
		static float v_coord[]=
		{
			+1.0000, +1.0000, +0.0000,
			-1.0000, +1.0000, +0.0000,
			-1.0000, -1.0000, +0.0000,
			+1.0000, +1.0000, +0.0000,
			-1.0000, -1.0000, +0.0000,
			+1.0000, -1.0000, +0.0000
			
		};

		static float t_coord[]=
		{
			1.0000, 0.0000,
			0.0000, 0.0000,
			0.0000, 1.0000,
			1.0000, 0.0000,
			0.0000, 1.0000,
			1.0000, 1.0000
		};
		GLuint attrIndex[2];
		
		//------------------------------------------------------------
		static GLuint textUnit;
		static GLuint VBO[2];
		
		if(flag == 0)
		{
			flag = 1;
			
			/* 產生Vexter Buffer Object. */
			glGenBuffers(2, &VBO[0]);
			glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(v_coord), v_coord, GL_STATIC_DRAW);
			
			glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(t_coord), t_coord, GL_STATIC_DRAW);
			
			/* 產生紋理物件. */
			glGenTextures(1, &textUnit);
			glBindTexture(GL_TEXTURE_2D, textUnit);
			/* 傳入影像到GPU */
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}
		else
		{
			/* 傳入影像到GPU */
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb);
		}
		
		/* 設定打畫面的大小. */
		glViewport(0, 0, screenWidth, screenHeight);
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(progImageShow.getProgram());
		
		/* 傳入頂點座標. */
		attrIndex[0] = glGetAttribLocation(progImageShow.getProgram(), "vPosition");
		glEnableVertexAttribArray(attrIndex[0]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
		glVertexAttribPointer(attrIndex[0], 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		
		/* 傳入紋理座標. */
		attrIndex[1] = glGetAttribLocation(progImageShow.getProgram(), "a_texCoord");
		glEnableVertexAttribArray(attrIndex[1]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
		glVertexAttribPointer(attrIndex[1], 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
		
		/* 把已經傳到GPU的紋理，傳到shader */
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textUnit);
		glUniform1i(glGetUniformLocation(progImageShow.getProgram(), "u_image"), 0);

		/* 打畫面 */
		glDrawArrays(GL_TRIANGLES, 0, 6);	
	}
#endif
