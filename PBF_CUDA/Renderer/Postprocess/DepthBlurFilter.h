#pragma once

#include <glm/glm.hpp>
#include "../Manager/FrameBuffer.h"

namespace Renderer
{
	class DepthBlurFilter
	{
	protected:
		int m_width, m_height;
		unsigned int m_screenQuadIndex;
		unsigned int m_blurShaderIndex;
		FrameBuffer::ptr m_framebuffer;

	public:
		typedef std::shared_ptr<DepthBlurFilter> ptr;

		DepthBlurFilter(int w, int h, unsigned int screenMesh);
		virtual ~DepthBlurFilter() = default;
		
		virtual unsigned int blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat) = 0;
	};

	class DepthGaussianBlurFilter : public DepthBlurFilter
	{
	public:
		typedef std::shared_ptr<DepthGaussianBlurFilter> ptr;

		DepthGaussianBlurFilter(int w, int h, unsigned int screenMesh);
		virtual ~DepthGaussianBlurFilter() = default;

		virtual unsigned int blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat);
	};

	class DepthBilateralSeperateBlurFilter : public DepthBlurFilter
	{
	public:
		typedef std::shared_ptr<DepthBilateralSeperateBlurFilter> ptr;

		DepthBilateralSeperateBlurFilter(int w, int h, unsigned int screenMesh);
		virtual ~DepthBilateralSeperateBlurFilter() = default;

		virtual unsigned int blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat);
	};

	class DepthBilateralCombineBlurFilter : public DepthBlurFilter
	{
	public:
		typedef std::shared_ptr<DepthBilateralCombineBlurFilter> ptr;

		DepthBilateralCombineBlurFilter(int w, int h, unsigned int screenMesh);
		virtual ~DepthBilateralCombineBlurFilter() = default;

		virtual unsigned int blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat);
	};

	class DepthCurvatureFlowBlurFilter : public DepthBlurFilter
	{
	private:
		unsigned int m_iterations;

	public:
		typedef std::shared_ptr<DepthCurvatureFlowBlurFilter> ptr;

		DepthCurvatureFlowBlurFilter(int w, int h, unsigned int screenMesh,
			unsigned int iters = 20);
		virtual ~DepthCurvatureFlowBlurFilter() = default;

		virtual unsigned int blurTexture(unsigned int targetTexIndex, const glm::mat4 &projectMat);
	};
}