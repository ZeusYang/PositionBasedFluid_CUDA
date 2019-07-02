#include "DepthBlurFilter.h"

#include "../Manager/MeshMgr.h"
#include "../Manager/Geometry.h"
#include "../Manager/ShaderMgr.h"
#include "../Manager/TextureMgr.h"

namespace Renderer
{
	DepthBlurFilter::DepthBlurFilter(int w, int h, unsigned int screenMesh)
		:m_width(w), m_height(h)
	{
		m_blurShaderIndex = -1;
		// screen quad.
		m_screenQuadIndex = screenMesh;
		m_framebuffer = std::shared_ptr<FrameBuffer>(
			new FrameBuffer(w, h, "depthBlur", {}, false));
	}

	DepthGaussianBlurFilter::DepthGaussianBlurFilter(int w, int h, unsigned int screenMesh)
		: DepthBlurFilter(w, h, screenMesh)
	{
		m_blurShaderIndex = ShaderMgr::getSingleton()->loadShader("DepthGaussianBlur",
			"./glsl/DepthGaussianBlur.vert", "./glsl/DepthGaussianBlur.frag");
	}

	unsigned int DepthGaussianBlurFilter::blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat)
	{
		m_framebuffer->bind();
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glClear(GL_DEPTH_BUFFER_BIT);

		Shader::ptr blurShader = ShaderMgr::getSingleton()->getShader(m_blurShaderIndex);
		blurShader->bind();
		blurShader->setInt("image", 0);
		TextureMgr::getSingleton()->bindTexture(targetTexIndex, 0);
		
		for (unsigned int index = 0; index < 5; ++index) {
			// horizontal blur.
			blurShader->setInt("horizontal", 1);
			MeshMgr::getSingleton()->drawMesh(m_screenQuadIndex, false, 0);

			// copy to target texture.
			glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_width, m_height);

			// vertical blur.
			blurShader->setInt("horizontal", 0);
			MeshMgr::getSingleton()->drawMesh(m_screenQuadIndex, false, 0);

			// copy to target texture.
			glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_width, m_height);
		}

		TextureMgr::getSingleton()->unBindTexture(targetTexIndex);
		blurShader->unBind();
		m_framebuffer->unBind();
		return targetTexIndex;
	}


	DepthBilateralSeperateBlurFilter::DepthBilateralSeperateBlurFilter(int w, int h, unsigned int screenMesh)
		:DepthBlurFilter(w, h, screenMesh)
	{
		m_blurShaderIndex = ShaderMgr::getSingleton()->loadShader("DepthBilateralSeperateBlur",
			"./glsl/DepthBilateralSeperateBlur.vert", "./glsl/DepthBilateralSeperateBlur.frag");
	}

	unsigned int DepthBilateralSeperateBlurFilter::blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat)
	{
		m_framebuffer->bind();
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glClear(GL_DEPTH_BUFFER_BIT);

		Shader::ptr blurShader = ShaderMgr::getSingleton()->getShader(m_blurShaderIndex);
		blurShader->bind();
		blurShader->setInt("image", 0);
		blurShader->setFloat("filterRadius", 6.0f);
		TextureMgr::getSingleton()->bindTexture(targetTexIndex, 0);

		for (unsigned int index = 0; index < 5; ++index) {
			// horizontal blur.
			blurShader->setVec2("blurDir", glm::vec2(1, 0));
			MeshMgr::getSingleton()->drawMesh(m_screenQuadIndex, false, 0);

			// copy to target texture.
			glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_width, m_height);

			// vertical blur.
			blurShader->setVec2("blurDir", glm::vec2(0, 1));
			MeshMgr::getSingleton()->drawMesh(m_screenQuadIndex, false, 0);

			// copy to target texture.
			glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_width, m_height);
		}

		TextureMgr::getSingleton()->unBindTexture(targetTexIndex);
		blurShader->unBind();
		m_framebuffer->unBind();
		return targetTexIndex;
	}

	DepthBilateralCombineBlurFilter::DepthBilateralCombineBlurFilter(int w, int h, unsigned int screenMesh)
		: DepthBlurFilter(w, h, screenMesh)
	{
		m_blurShaderIndex = ShaderMgr::getSingleton()->loadShader("DepthBilateralCombineBlur",
			"./glsl/DepthBilateralCombineBlur.vert", "./glsl/DepthBilateralCombineBlur.frag");
	}

	unsigned int DepthBilateralCombineBlurFilter::blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat)
	{
		m_framebuffer->bind();
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glClear(GL_DEPTH_BUFFER_BIT);

		// blur.
		Shader::ptr blurShader = ShaderMgr::getSingleton()->getShader(m_blurShaderIndex);
		blurShader->bind();
		blurShader->setInt("image", 0);
		blurShader->setFloat("filterRadius", 10.0f);
		TextureMgr::getSingleton()->bindTexture(targetTexIndex, 0);
		MeshMgr::getSingleton()->drawMesh(m_screenQuadIndex, false, 0);

		// copy to target texture.
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_width, m_height);
		TextureMgr::getSingleton()->unBindTexture(targetTexIndex);

		blurShader->unBind();
		m_framebuffer->unBind();
		return targetTexIndex;
	}

	DepthCurvatureFlowBlurFilter::DepthCurvatureFlowBlurFilter(int w, int h,
		unsigned int screenMesh, unsigned int iters)
		:DepthBlurFilter(w, h, screenMesh)
	{
		m_iterations = iters;
		m_blurShaderIndex = ShaderMgr::getSingleton()->loadShader("DepthCurvatureFlowBlur",
			"./glsl/DepthCurvatureFlowBlur.vert", "./glsl/DepthCurvatureFlowBlur.frag");
	}

	unsigned int DepthCurvatureFlowBlurFilter::blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat)
	{
		m_framebuffer->bind();
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glClear(GL_DEPTH_BUFFER_BIT);

		// blur.
		Shader::ptr blurShader = ShaderMgr::getSingleton()->getShader(m_blurShaderIndex);
		blurShader->bind();
		blurShader->setInt("image", 0);
		blurShader->setFloat("step", 0.00065f);
		blurShader->setFloat("step", 0.00070f);
		blurShader->setMat4("projectMatrix", projectMat);
		TextureMgr::getSingleton()->bindTexture(targetTexIndex, 0);

		for (unsigned int iter = 0; iter < m_iterations; ++iter)
		{
			// blur.
			MeshMgr::getSingleton()->drawMesh(m_screenQuadIndex, false, 0);

			// copy to target texture.
			glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_width, m_height);
		}

		TextureMgr::getSingleton()->unBindTexture(targetTexIndex);
		blurShader->unBind();
		m_framebuffer->unBind();
		return targetTexIndex;
	}

}