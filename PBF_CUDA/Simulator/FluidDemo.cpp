#include "FluidDemo.h"

#include <vector>
#include <iostream>

using namespace Simulator;

FluidDemo::FluidDemo()
{
	m_simulator = nullptr;
}

void FluidDemo::fluidDeamBreak(unsigned int gridSize, float length, float radius)
{
	unsigned int numParticles = 0;
	float spacing = radius * 2.0f;
	std::vector<float> positions;
	std::vector<float> velocities;
	float jitter = radius * 0.01f;
	srand(1973);

	// bottom fluid.
	glm::vec3 bottomFluidSize = glm::vec3(30.0f, 30.0f, 30.0f);
	glm::ivec3 bottomFluidDim = glm::ivec3(bottomFluidSize.x / spacing,
		bottomFluidSize.y / spacing, bottomFluidSize.z / spacing);
	for (int z = 0; z < bottomFluidDim.z; ++z)
	{
		for (int y = 0; y < bottomFluidDim.y; ++y)
		{
			for (int x = 0; x < bottomFluidDim.x; ++x)
			{
				positions.push_back(spacing*x + radius - 0.5f * 80.0f + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(spacing*y + radius - 0.5f * 40.0f + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(spacing*z + radius - 0.5f * 40.0f + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(1.0f);

				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
			}
		}
	}

	numParticles = positions.size() / 4;
	m_simulator = std::shared_ptr<FluidSystem>(new FluidSystem(numParticles,
		{ gridSize, gridSize, gridSize }, radius));
	m_simulator->setParticlePositions(&positions[0], 0, numParticles);
	m_simulator->setParticleVelocities(&velocities[0], 0, numParticles);
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
					positions.push_back(spacing*x + radius - 0.5f * length + (frand()*2.0f - 1.0f)*jitter);
					positions.push_back(spacing*y + radius - 0.5f * length + (frand()*2.0f - 1.0f)*jitter);
					positions.push_back(spacing*z + radius - 0.5f * length + (frand()*2.0f - 1.0f)*jitter);
					positions.push_back(1.0f);

					velocities.push_back(0.0f);
					velocities.push_back(0.0f);
					velocities.push_back(0.0f);
					velocities.push_back(0.0f);
				}
			}
		}
	}
	m_simulator->setParticlePositions(&positions[0], 0, numParticles);
	m_simulator->setParticleVelocities(&velocities[0], 0, numParticles);
}

void FluidDemo::sphereFluidDrop(unsigned int gridSize, float length, float radius)
{
	unsigned int numParticles = 0;
	float spacing = radius * 2.0f;
	std::vector<float> positions;
	std::vector<float> velocities;
	float jitter = radius * 0.01f;
	srand(1973);

	// bottom fluid.
	glm::vec3 bottomFluidSize = glm::vec3(80.0f, 5.0f, 40.0f);
	glm::ivec3 bottomFluidDim = glm::ivec3(bottomFluidSize.x / spacing,
		bottomFluidSize.y / spacing, bottomFluidSize.z / spacing);
	for (int z = 0; z < bottomFluidDim.z; ++z)
	{
		for (int y = 0; y < bottomFluidDim.y; ++y)
		{
			for (int x = 0; x < bottomFluidDim.x; ++x)
			{
				positions.push_back(spacing*x + radius - 0.5f * bottomFluidSize.x + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(spacing*y + radius - 0.5f * 40.0f + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(spacing*z + radius - 0.5f * bottomFluidSize.z + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(1.0f);

				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
			}
		}
	}

	// sphere fluid.
	float spherRadius = 8.0f;
	float boxLength = 2.0f * spherRadius;
	int boxDim = boxLength / spacing;
	for (int z = 0; z < boxDim; ++z)
	{
		for (int y = 0; y < boxDim; ++y)
		{
			for (int x = 0; x < boxDim; ++x)
			{
				float dx = x * spacing - 0.5f * boxLength;
				float dy = y * spacing - 0.5f * boxLength;
				float dz = z * spacing - 0.5f * boxLength;
				float l = sqrtf(dx * dx + dy * dy + dz * dz);
				if (l > spherRadius) continue;

				positions.push_back(dx + radius + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(dy + radius + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(dz + radius + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(1.0f);

				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
			}
		}
	}

	numParticles = positions.size() / 4;
	m_simulator = std::shared_ptr<FluidSystem>(new FluidSystem(numParticles,
		{ gridSize, gridSize, gridSize }, radius));
	m_simulator->setParticlePositions(&positions[0], 0, numParticles);
	m_simulator->setParticleVelocities(&velocities[0], 0, numParticles);
}

void FluidDemo::addSphereFluid(glm::vec3 center, float radius)
{
	if (m_simulator == nullptr)
		return;
	std::vector<float> positions;
	std::vector<float> velocities;

	SimulateParams &params = m_simulator->getSimulateParams();
	float spacing = params.m_particleRadius * 2.0f;
	float jitter = params.m_particleRadius * 0.01f;
	srand(1973);
	int count = 0;
	float length = 2.0 * radius;
	int resolution = length / (2.0f * params.m_particleRadius);
	for (int z = 0; z < resolution; ++z)
	{
		for (int y = 0; y < resolution; ++y)
		{
			for (int x = 0; x < resolution; ++x)
			{
				float dx = x * spacing - 0.5f * length;
				float dy = y * spacing - 0.5f * length;
				float dz = z * spacing - 0.5f * length;
				float l = sqrtf(dx * dx + dy * dy + dz * dz);
				if (l > length) continue;
				positions.push_back(dx + center.x + params.m_particleRadius + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(dy + center.y + params.m_particleRadius + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(dz + center.z + params.m_particleRadius + (frand()*2.0f - 1.0f)*jitter);
				positions.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				velocities.push_back(0.0f);
				++count;
			}
		}
	}
	m_simulator->addParticles(positions, velocities, count);
}
