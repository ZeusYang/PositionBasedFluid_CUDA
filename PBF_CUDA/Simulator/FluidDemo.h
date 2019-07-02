#pragma once

#include "FluidSystem.h"
#include <glm/glm.hpp>

class FluidDemo
{
private:
	Simulator::FluidSystem::ptr m_simulator;

public:
	FluidDemo();
	~FluidDemo() = default;

	Simulator::FluidSystem::ptr getSimulator() { return m_simulator; }

	void fluidDeamBreak(unsigned int gridSize, float length, float radius);
	void boxFluidDrop(unsigned int numParticles, unsigned int gridSize, float length, float radius);
	void sphereFluidDrop(unsigned int gridSize, float length, float radius);

	void addSphereFluid(glm::vec3 center, float radius);

private:
	inline float frand()
	{
		return rand() / (float)RAND_MAX;
	}
};

