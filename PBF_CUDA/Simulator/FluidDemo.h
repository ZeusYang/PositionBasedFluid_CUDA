#pragma once

#include "FluidSystem.h"

class FluidDemo
{
private:
	Simulator::FluidSystem::ptr m_simulator;

public:
	FluidDemo();
	~FluidDemo() = default;

	Simulator::FluidSystem::ptr getSimulator() { return m_simulator; }

	void boxFluidDrop(unsigned int numParticles, unsigned int gridSize, float length, float radius);
private:
	inline float frand()
	{
		return rand() / (float)RAND_MAX;
	}
};

