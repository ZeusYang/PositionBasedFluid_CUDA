#pragma once

#include <vector>
#include <memory>
#include <driver_types.h>
#include "SimulateParams.cuh"

namespace Simulator
{
	class FluidSystem
	{
	private:
		bool m_initialized;
		float *m_devicePos;			// position, density.
		float *m_deviceVel;			// velocity, lambda.
		float *m_deviceDeltaPos;
		float *m_devicePredictedPos;
		unsigned int *m_deviceCellStart;
		unsigned int *m_deviceCellEnd;
		unsigned int *m_deviceGridParticleHash;
		
		unsigned int m_posVBO;
		SimulateParams m_params;
		cudaGraphicsResource *m_cudaPosVBORes;

	public:
		typedef std::shared_ptr<FluidSystem> ptr;

		FluidSystem(unsigned int numParticles, uint3 gridSize, float radius);
		~FluidSystem();

		void simulate(float deltaTime);

		unsigned int getPosVBO()const { return m_posVBO; }
		SimulateParams &getSimulateParams() { return m_params; }
		void setResetDensity(const float &value);
		void setParticlePositions(const float *data, int start, int nums);
		void setParticleVelocities(const float *data, int start, int nums);

		void addParticles(const std::vector<float> &pos, const std::vector<float> &vel, unsigned int num);

	public:
		void initialize(int numParticles);
		void finalize();
	};
}
