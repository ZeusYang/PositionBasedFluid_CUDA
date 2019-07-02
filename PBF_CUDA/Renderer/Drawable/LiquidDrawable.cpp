#include "LiquidDrawable.h"

#include "../Manager/MeshMgr.h"

namespace Renderer
{
	LiquidDrawable::LiquidDrawable(unsigned int scrWidth, unsigned int scrHei)
		:m_screenWidth(scrWidth), m_screenHeight(scrHei)
	{
		// screen quad mesh.
		m_screenQuadIndex = MeshMgr::getSingleton()->loadMesh(new ScreenQuad());

		// initialize.
		m_backgroundTex = -1;
		m_backgroundDepthTex = -1;
		m_renderTarget = 0;
		m_liquidColor = glm::vec4(.275f, 0.65f, 0.85f, 0.5f);
		m_depthBlurFilter = std::shared_ptr<DepthBilateralCombineBlurFilter>(new
			DepthBilateralCombineBlurFilter(scrWidth, scrHei, m_screenQuadIndex));

		// initialize
		m_particleVBO = 0;
		m_numParticles = 0;
		m_particleRadius = 1.0f;
		m_densityLowerBound = 1.0f / (8.0f * pow(m_particleRadius, 3.0f)) * 0.01f;
		glGenVertexArrays(1, &m_particleVAO);

		// load shader and texture.
		m_shaderMgr = ShaderMgr::getSingleton();
		m_textureMgr = TextureMgr::getSingleton();
		m_shaderIndex = m_shaderMgr->loadShader("liquidDepth",
			"./glsl/liquidDepth.vert", "./glsl/liquidDepth.frag");
		m_shaderMgr->loadShader("liquidThick",
			"./glsl/liquidThickness.vert", "./glsl/liquidThickness.frag");
		m_shaderMgr->loadShader("liquidRender",
			"./glsl/liquidRender.vert", "./glsl/liquidRender.frag");

		// framebuffer.
		m_framebuffer = std::shared_ptr<FrameBuffer>(
			new FrameBuffer(m_screenWidth, m_screenHeight, "LiquidDepth", { "LiquidThickness" }, true));
	}

	LiquidDrawable::~LiquidDrawable()
	{
		glDeleteVertexArrays(1, &m_particleVAO);
	}

	void LiquidDrawable::setLiquidBlur(int blur)
	{
		switch (blur)
		{
		case 0:// Bilateral Combined Blur.
			m_depthBlurFilter = std::shared_ptr<DepthBilateralCombineBlurFilter>(new
				DepthBilateralCombineBlurFilter(m_screenWidth, m_screenHeight, m_screenQuadIndex));
			break;
		case 1:// Bilateral Seperated Blur.
			m_depthBlurFilter = std::shared_ptr<DepthBilateralSeperateBlurFilter>(new
				DepthBilateralSeperateBlurFilter(m_screenWidth, m_screenHeight, m_screenQuadIndex));
			break;
		case 2:// Gaussian Blur.
			m_depthBlurFilter = std::shared_ptr<DepthGaussianBlurFilter>(new
				DepthGaussianBlurFilter(m_screenWidth, m_screenHeight, m_screenQuadIndex));
			break;
		case 3:// Curvature Flow Blur.
			m_depthBlurFilter = std::shared_ptr<DepthCurvatureFlowBlurFilter>(new
				DepthCurvatureFlowBlurFilter(m_screenWidth, m_screenHeight, m_screenQuadIndex, 50));
			break;
		}
	}

	void LiquidDrawable::setLiquidColor(glm::vec4 color)
	{
		m_liquidColor = color;
	}

	void LiquidDrawable::setParticleRadius(float radius)
	{
		m_particleRadius = radius;
		m_densityLowerBound = 1.0f / (8.0f * pow(m_particleRadius, 3.0f)) * 0.001f;
	}

	void LiquidDrawable::setParticleVBO(unsigned int vbo, int numParticles)
	{
		m_particleVBO = vbo;
		m_numParticles = numParticles;

		glBindVertexArray(m_particleVAO);
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
			static_cast<void*>(0));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void LiquidDrawable::setBackgroundTexAndRenderTarget(unsigned int rTex, unsigned int rDepth, unsigned int rTar)
	{
		m_backgroundTex = rTex;
		m_backgroundDepthTex = rDepth;
		m_renderTarget = rTar;
	}

	void LiquidDrawable::render(Camera3D::ptr camera, Light::ptr sunLight,
		Camera3D::ptr lightCamera, Shader::ptr shader)
	{
		if (!m_visiable) return;

		// draw liquid thickness.
		drawLiquidThick(camera, sunLight, lightCamera, shader);

		// draw liquid depth.
		drawLiquidDepth(camera, sunLight, lightCamera, shader);

		// depth blur.
		m_depthBlurFilter->blurTexture(m_framebuffer->getDepthTexIndex(), camera->getProjectMatrix());

		// thickness blur.
		//m_depthBlurFilter->blurTexture(m_framebuffer->getColorTexIndex(0));

		// draw liquid surface.
		{
			glBindFramebuffer(GL_FRAMEBUFFER, m_renderTarget);
			glViewport(0, 0, m_screenWidth, m_screenHeight);
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_BLEND);

			shader = m_shaderMgr->getShader("liquidRender");
			shader->bind();
			if (sunLight)
				sunLight->setLightUniform(shader, camera);
			shader->setInt("depthTex", 0);
			shader->setInt("thicknessTex", 1);
			shader->setInt("backgroundTex", 2);
			shader->setInt("backgroundDepthTex", 3);
			shader->setVec4("liquidColor", m_liquidColor);
			shader->setMat4("viewMatrix", camera->getViewMatrix());
			shader->setMat4("invViewMatrix", camera->getInvViewMatrix());
			shader->setMat4("invProjectMatrix", camera->getInvProjectMatrix());
			TextureMgr::getSingleton()->bindTexture("LiquidDepth", 0);
			TextureMgr::getSingleton()->bindTexture("LiquidThickness", 1);
			TextureMgr::getSingleton()->getTexture(m_backgroundTex)->bind(2);
			TextureMgr::getSingleton()->getTexture(m_backgroundDepthTex)->bind(3);

			MeshMgr::getSingleton()->drawMesh(m_screenQuadIndex, false, 0);

			TextureMgr::getSingleton()->unBindTexture("LiquidDepth");
			TextureMgr::getSingleton()->unBindTexture("LiquidThickness");
			TextureMgr::getSingleton()->getTexture(m_backgroundTex)->unBind();
			shader->unBind();
		}
	}

	void LiquidDrawable::renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera)
	{
		return;
	}

	void LiquidDrawable::drawLiquidDepth(Camera3D::ptr camera, Light::ptr sunLight,
		Camera3D::ptr lightCamera, Shader::ptr shader)
	{
		m_framebuffer->bind();

		// calculate particle size scale factor.
		float aspect = camera->getAspect();
		float fovy = camera->getFovy();
		float pointScale = 1.0f * m_screenWidth / aspect * (1.0f / tanf(glm::radians(fovy) * 0.5f));
		
		// render state.
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		// shader.
		shader = m_shaderMgr->getShader("liquidDepth");
		shader->bind();
		shader->setFloat("farPlane", camera->getFar());
		shader->setFloat("nearPlane", camera->getNear());
		shader->setFloat("pointScale", pointScale);
		shader->setFloat("pointSize", m_particleRadius);
		shader->setFloat("densityLowerBound", m_densityLowerBound);
		shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
		shader->setMat4("viewMatrix", camera->getViewMatrix());
		shader->setMat4("projectMatrix", camera->getProjectMatrix());

		// draw
		glBindVertexArray(m_particleVAO);
		glDrawArrays(GL_POINTS, 0, m_numParticles);
		glBindVertexArray(0);

		// restore.
		m_shaderMgr->unBindShader();
		glDisable(GL_PROGRAM_POINT_SIZE);
		m_framebuffer->unBind();

	}

	void LiquidDrawable::drawLiquidThick(Camera3D::ptr camera, Light::ptr sunLight,
		Camera3D::ptr lightCamera, Shader::ptr shader)
	{
		m_framebuffer->bind();

		// calculate particle size scale factor.
		float aspect = camera->getAspect();
		float fovy = camera->getFovy();
		float pointScale = 1.0f * m_screenWidth / aspect * (1.0f / tanf(glm::radians(fovy) * 0.5f));

		// render state.
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		glDepthMask(GL_FALSE);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);

		// shader.
		shader = m_shaderMgr->getShader("liquidThick");
		shader->bind();
		shader->setFloat("pointScale", pointScale);
		shader->setFloat("pointSize", 4.0f * m_particleRadius);
		shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
		shader->setMat4("viewMatrix", camera->getViewMatrix());
		shader->setMat4("projectMatrix", camera->getProjectMatrix());

		// draw
		glBindVertexArray(m_particleVAO);
		glDrawArrays(GL_POINTS, 0, m_numParticles);
		glBindVertexArray(0);

		// restore.
		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
		glDisable(GL_PROGRAM_POINT_SIZE);
		m_shaderMgr->unBindShader();
		m_framebuffer->unBind();
	}

}