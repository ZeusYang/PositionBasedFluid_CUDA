#include "FluidDemo.h"

#include <vector>
#include <iostream>

using namespace Simulator;

FluidDemo::FluidDemo()
{
	m_simulator = nullptr;
}

void FluidDemo::boxFluidDrop(unsigned int numParticles, unsigned int gridSize, float length, float radius)
{
	m_simulator = std::shared_ptr<FluidSystem>(new FluidSystem(numParticles, { gridSize, gridSize, gridSize }, radius));
	SimulateParams &params = m_simulator->getSimulateParams();
	float spacing = params.m_particleRadius * 2.0f;
	unsigned int fluidSize = (int)ceilf(powf((float)numParticles, 1.0f / 3.0f));
	std::vector<float> positions;
	std::vector<float> velocities;
	length = fluidSize * spacing;
	float jitter = params.m_particleRadius * 0.01f;
	srand(1973);
	for (unsigned int z = 0; z < fluidSize; ++z)
	{
		for (unsigned int y = 0; y < fluidSize; ++y)
		{
			for (unsigned int x = 0; x < fluidSize; ++x)
			{
				unsigned int index = z * fluidSize * fluidSize + y * fluidSize + x;
				if (index < numParticles)
				{
					positions.push_back(spacing*x + params.m_particleRadius - 0.5f * length + (frand()*2.0f - 1.0f)*jitter);
					positions.push_back(spacing*y + params.m_particleRadius - 0.5f * length + (frand()*2.0f - 1.0f)*jitter);
					positions.push_back(spacing*z + params.m_particleRadius - 0.5f * length + (frand()*2.0f - 1.0f)*jitter);
					positions.push_back(1.0f);

					velocities.push_back(0.0f);
					velocities.push_back(0.0f);
					velocities.push_back(0.0f);
					velocities.push_back(0.0f);
				}
			}
		}
	}
	m_simulator->setParticlePositions(&positions[0], numParticles);
	m_simulator->setParticleVelocities(&velocities[0], numParticles);

}
