#include "FluidSystem.h"

#include <iostream>
#include <GL/glew.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <cuda_gl_interop.h>

extern void getLastCudaError(const char *errorMessage);
extern void setParameters(SimulateParams *hostParams);
extern void computeHash(
	unsigned int *gridParticleHash,
	float *pos,
	int numParticles);
extern void sortParticles(
	unsigned int *deviceGridParticleHash,
	unsigned int numParticles,
	float *devicePos,
	float *deviceVel,
	float *devicePredictedPos);
extern void findCellRange(
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCell);
extern void fluidAdvection(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles);
extern void densityConstraint(
	float4 *position,
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells);
extern void updateVelAndPos(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles);
extern void applyXSPHViscosity(
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	float deltaTime,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles);
extern void applyVorticityConfinment(
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	float deltaTime,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles);

namespace Simulator
{
#define M_PI 3.1415926535
	FluidSystem::FluidSystem(unsigned int numParticles, uint3 gridSize, float radius) :
		m_initialized(false),
		m_devicePos(nullptr),
		m_deviceVel(nullptr),
		m_deviceDeltaPos(nullptr),
		m_devicePredictedPos(nullptr)
	{
		// particles and grid.
		m_params.m_gridSize = gridSize;
		m_params.m_particleRadius = radius;
		m_params.m_numParticles = numParticles;
		m_params.m_numGridCells = gridSize.x * gridSize.y * gridSize.z;

		// iteration number.
		m_params.m_maxIterNums = 3;

		// smooth kernel radius.
		m_params.m_sphRadius = 4 * radius;
		m_params.m_sphRadiusSquared = m_params.m_sphRadius * m_params.m_sphRadius;

		// lagrange multiplier eps.
		m_params.m_lambdaEps = 1000.0f;
		// vorticity force coff.
		m_params.m_vorticity = 0.001f;
		// viscosity force coff.
		m_params.m_viscosity = 0.01f;

		// left boundary wall.
		m_params.m_leftWall = -40.0f;

		// fluid reset density.
		m_params.m_restDensity = 1.0f / (8.0f * powf(radius, 3.0f));
		m_params.m_invRestDensity = 1.0f / m_params.m_restDensity;
		
		// sph kernel function coff.
		m_params.m_poly6Coff = 315.0f / (64.0f * M_PI * powf(m_params.m_sphRadius, 9.0));
		m_params.m_spikyGradCoff = -45.0f / (M_PI * powf(m_params.m_sphRadius, 6.0));

		// grid cells.
		float cellSize = 4.0f * m_params.m_particleRadius;
		m_params.m_cellSize = make_float3(cellSize, cellSize, cellSize);
		m_params.m_gravity = make_float3(0.0f, -9.8f, 0.0f);
		m_params.m_worldOrigin = { -42.0f,-22.0f,-22.0f };

		m_params.m_oneDivWPoly6 = 1.0f / (m_params.m_poly6Coff *
			pow(m_params.m_sphRadiusSquared - pow(0.1 * m_params.m_sphRadius, 2.0), 3.0));

		initialize(numParticles);
	}

	FluidSystem::~FluidSystem()
	{
		//finalize();
	}

	void FluidSystem::simulate(float deltaTime)
	{
		if (!m_initialized)
		{
			std::cout << "Must initialized first.\n";
			return;
		}

		size_t numBytes = 0;
		cudaGraphicsMapResources(1, &m_cudaPosVBORes, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&m_devicePos, &numBytes, m_cudaPosVBORes);

		// update constants
		setParameters(&m_params);

		// advection.
		{
			fluidAdvection(
				(float4*)m_devicePos,
				(float4*)m_deviceVel,
				(float4*)m_devicePredictedPos,
				deltaTime,
				m_params.m_numParticles);
			getLastCudaError("fluidAdvection");
		}

		// find neighbourhood.
		{
			// calculate grid Hash.
			computeHash(
				m_deviceGridParticleHash,
				m_devicePredictedPos,
				m_params.m_numParticles
			);
			getLastCudaError("computeHash");

			// sort particles based on hash value.
			sortParticles(
				m_deviceGridParticleHash,
				m_params.m_numParticles,
				m_devicePos,
				m_deviceVel,
				m_devicePredictedPos);
			getLastCudaError("sortParticles");

			// find start index and end index of each cell.
			findCellRange(
				m_deviceCellStart,
				m_deviceCellEnd,
				m_deviceGridParticleHash,
				m_params.m_numParticles,
				m_params.m_numGridCells);
			getLastCudaError("findCellRange");
		}

		// density constraint.
		unsigned int iter = 0;
		while (iter < m_params.m_maxIterNums)
		{
			densityConstraint(
				(float4*)m_devicePos,
				(float4*)m_deviceVel,
				(float3*)m_deviceDeltaPos,
				(float4*)m_devicePredictedPos,
				m_deviceCellStart,
				m_deviceCellEnd,
				m_deviceGridParticleHash,
				m_params.m_numParticles,
				m_params.m_numGridCells);
			++iter;
		}

		// update velocity and position.
		{
			updateVelAndPos(
				(float4*)m_devicePos,
				(float4*)m_deviceVel,
				(float4*)m_devicePredictedPos,
				deltaTime,
				m_params.m_numParticles);
		}

		// apply vorticity confinment.
		//{
		//	applyVorticityConfinment(
		//		(float4*)m_deviceVel,
		//		(float3*)m_deviceDeltaPos,
		//		(float4*)m_devicePredictedPos,
		//		deltaTime,
		//		m_deviceCellStart,
		//		m_deviceCellEnd,
		//		m_deviceGridParticleHash,
		//		m_params.m_numParticles);
		//}

		// apply XSPH viscosity.
		//{
		//	applyXSPHViscosity(
		//		(float4*)m_deviceVel,
		//		(float3*)m_deviceDeltaPos,
		//		(float4*)m_devicePredictedPos,
		//		deltaTime,
		//		m_deviceCellStart,
		//		m_deviceCellEnd,
		//		m_deviceGridParticleHash,
		//		m_params.m_numParticles);
		//}

		cudaGraphicsUnmapResources(1, &m_cudaPosVBORes, 0);
	}

	void FluidSystem::setResetDensity(const float & value)
	{
		m_params.m_restDensity = value;
		m_params.m_invRestDensity = 1.0f / value;
	}

	void FluidSystem::setParticlePositions(const float * data, int nums)
	{
		cudaGraphicsUnregisterResource(m_cudaPosVBORes);
		getLastCudaError("setParticlePositions.cudaGraphicsUnregisterResource");
		glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, nums * 4 * sizeof(float), data);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&m_cudaPosVBORes, m_posVBO, cudaGraphicsMapFlagsNone);
		getLastCudaError("setParticlePositions.cudaGraphicsGLRegisterBuffer");
	}

	void FluidSystem::setParticleVelocities(const float * data, int nums)
	{
		cudaMemcpy((char*)m_deviceVel, data, nums * 4 * sizeof(float), cudaMemcpyHostToDevice);
		getLastCudaError("setParticleVelocities");
	}

	void FluidSystem::initialize(int numParticles)
	{
		if (m_initialized)
		{
			std::cout << "Already initialized.\n";
			return;
		}

		m_params.m_numParticles = numParticles;
		unsigned int memSize = sizeof(float) * 4 * m_params.m_numParticles;

		// vbo.
		glGenBuffers(1, &m_posVBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
		glBufferData(GL_ARRAY_BUFFER, memSize, 0, GL_DYNAMIC_DRAW);
		int size = 0;
		glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint *)&size);
		if ((unsigned)size != memSize)
			fprintf(stderr, "WARNING: Pixel Buffer Object allocation failed!\n");
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&m_cudaPosVBORes, m_posVBO, cudaGraphicsMapFlagsNone);
		getLastCudaError("cudaGraphicsGLRegisterBuffer");

		// allocation.
		cudaMalloc((void**)&m_deviceVel, memSize);
		getLastCudaError("allocation1");
		cudaMalloc((void**)&m_devicePredictedPos, memSize);
		cudaMalloc((void**)&m_deviceDeltaPos, sizeof(float) * 3 * m_params.m_numParticles);
		cudaMalloc((void**)&m_deviceGridParticleHash, m_params.m_numParticles * sizeof(unsigned int));
		cudaMalloc((void**)&m_deviceCellStart, m_params.m_numGridCells * sizeof(unsigned int));
		cudaMalloc((void**)&m_deviceCellEnd, m_params.m_numGridCells * sizeof(unsigned int));
		getLastCudaError("allocation");

		setParameters(&m_params);
		getLastCudaError("setParameters");
		m_initialized = true;
	}

	void FluidSystem::finalize()
	{
		if (!m_initialized)
		{
			std::cout << "Haven't initialized.\n";
			return;
		}

		// free.
		cudaFree(m_deviceVel);
		getLastCudaError("free allocation.m_deviceVel");
		cudaFree(m_deviceDeltaPos);
		getLastCudaError("free allocation.m_deviceDeltaPos");
		cudaFree(m_devicePredictedPos);
		getLastCudaError("free allocation.m_devicePredictedPos");
		cudaFree(m_deviceGridParticleHash);
		getLastCudaError("free allocation.m_deviceGridParticleHash");
		cudaFree(m_deviceCellStart);
		getLastCudaError("free allocation.m_deviceCellStart");
		cudaFree(m_deviceCellEnd);
		getLastCudaError("free allocation.m_deviceCellEnd");

		// unregister.
		cudaGraphicsUnregisterResource(m_cudaPosVBORes);
		getLastCudaError("cudaGraphicsUnregisterResource.");
		glDeleteBuffers(1, &m_posVBO);
		m_initialized = false;
	}
}