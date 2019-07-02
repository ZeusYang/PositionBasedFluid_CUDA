#pragma once

#include "Drawable.h"
#include "../Manager/ShaderMgr.h"
#include "../Manager/TextureMgr.h"
#include "../Manager/FrameBuffer.h"
#include "../Postprocess/DepthBlurFilter.h"

namespace Renderer
{
	class LiquidDrawable : public Drawable
	{
	private:
		glm::vec4 m_liquidColor;
		float m_particleRadius;
		float m_densityLowerBound;
		unsigned int m_particleVAO;
		unsigned int m_particleVBO;
		unsigned int m_numParticles;
		unsigned int m_screenWidth;
		unsigned int m_screenHeight;
		unsigned int m_screenQuadIndex;
		ShaderMgr::ptr m_shaderMgr;
		TextureMgr::ptr m_textureMgr;
		FrameBuffer::ptr m_framebuffer;
		unsigned int m_backgroundTex;		// background refracted texture.
		unsigned int m_backgroundDepthTex;	// background refracted texture.
		unsigned int m_renderTarget;		// render buffer.
		DepthBlurFilter::ptr m_depthBlurFilter;

	public:
		typedef std::shared_ptr<LiquidDrawable> ptr;

		LiquidDrawable(unsigned int scrWidth, unsigned int scrHei);
		~LiquidDrawable();

		void setLiquidBlur(int blur);
		void setLiquidColor(glm::vec4 color);
		void setParticleRadius(float radius);
		void setParticleVBO(unsigned int vbo, int numParticles);
		void setBackgroundTexAndRenderTarget(unsigned int rTex, unsigned int rDepth, unsigned int rTar);

		virtual void render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera,
			Shader::ptr shader = nullptr);

		virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera);

	private:
		void drawLiquidDepth(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera,
			Shader::ptr shader = nullptr);
		void drawLiquidThick(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera,
			Shader::ptr shader = nullptr);
	};

}
