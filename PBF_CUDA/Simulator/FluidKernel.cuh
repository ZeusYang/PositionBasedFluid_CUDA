#ifndef _FLUID_KERNEL_H
#define _FLUID_KERNEL_H

#include <math.h>
#include <cooperative_groups.h>
#include <thrust/tuple.h>

#include "math_constants.h"
#include "SimulateParams.cuh"

using namespace cooperative_groups;

__constant__ SimulateParams params;

inline __host__ __device__ 
float3 cross(float3 a, float3 b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__
float wPoly6(const float3 &r)
{
	const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
	if (lengthSquared > params.m_sphRadiusSquared || lengthSquared <= 0.00000001f)
		return 0.0f;
	float iterm = params.m_sphRadiusSquared - lengthSquared;
	return params.m_poly6Coff * iterm * iterm * iterm;
}

__device__
float3 wSpikyGrad(const float3 &r)
{
	const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
	float3 ret = { 0.0f, 0.0f, 0.0f };
	if (lengthSquared > params.m_sphRadiusSquared || lengthSquared <= 0.00000001f)
		return ret;
	const float length = sqrtf(lengthSquared);
	float iterm = params.m_sphRadius - length;
	float coff = params.m_spikyGradCoff * iterm * iterm / length;
	ret.x = coff * r.x;
	ret.y = coff * r.y;
	ret.z = coff * r.z;
	return ret;
}

__device__
int3 calcGridPosKernel(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.m_worldOrigin.x) / params.m_cellSize.x);
	gridPos.y = floor((p.y - params.m_worldOrigin.y) / params.m_cellSize.y);
	gridPos.z = floor((p.z - params.m_worldOrigin.z) / params.m_cellSize.z);
	return gridPos;
}

__device__
unsigned int calcGridHashKernel(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.m_gridSize.x - 1);
	gridPos.y = gridPos.y & (params.m_gridSize.y - 1);
	gridPos.z = gridPos.z & (params.m_gridSize.z - 1);
	return gridPos.z * params.m_gridSize.x * params.m_gridSize.y + gridPos.y * params.m_gridSize.x + gridPos.x;
}

__global__
void calcParticlesHashKernel(
	unsigned int *gridParticleHash,
	float4 *pos,
	unsigned int numParticles
)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	volatile float4 curPos = pos[index];
	int3 gridPos = calcGridPosKernel(make_float3(curPos.x, curPos.y, curPos.z));
	unsigned int hashValue = calcGridHashKernel(gridPos);
	gridParticleHash[index] = hashValue;
}

__global__
void findCellRangeKernel(
	unsigned int *cellStart,			// output: cell start index
	unsigned int *cellEnd,				// output: cell end index
	unsigned int *gridParticleHash,		// input: sorted grid hashes
	unsigned int numParticles)
{
	thread_block cta = this_thread_block();
	extern __shared__ unsigned int sharedHash[];
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int hashValue;

	if (index < numParticles)
	{
		hashValue = gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hashValue;

		// first thread in block must load neighbor particle hash
		if (index > 0 && threadIdx.x == 0)
			sharedHash[0] = gridParticleHash[index - 1];
	}

	sync(cta);

	if (index < numParticles)
	{
		if (index == 0 || hashValue != sharedHash[threadIdx.x])
		{
			cellStart[hashValue] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
			cellEnd[hashValue] = index + 1;
	}
}

__global__
void advect(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readVel = velocity[index];
	float4 readPos = position[index];
	float3 nVel;
	float3 nPos;
	nVel.x = readVel.x + deltaTime * params.m_gravity.x;
	nVel.y = readVel.y + deltaTime * params.m_gravity.y;
	nVel.z = readVel.z + deltaTime * params.m_gravity.z;
	nPos.x = readPos.x + deltaTime * nVel.x;
	nPos.y = readPos.y + deltaTime * nVel.y;
	nPos.z = readPos.z + deltaTime * nVel.z;

	// collision with walls.
	if (nPos.x > 40.0f - params.m_particleRadius)
		nPos.x = 40.0f - params.m_particleRadius;
	if (nPos.x < params.m_leftWall + params.m_particleRadius)
		nPos.x = params.m_leftWall + params.m_particleRadius;

	if (nPos.y > 20.0f - params.m_particleRadius)
		nPos.y = 20.0f - params.m_particleRadius;
	if (nPos.y < -20.0f + params.m_particleRadius)
		nPos.y = -20.0f + params.m_particleRadius;

	if (nPos.z > 20.0f - params.m_particleRadius)
		nPos.z = 20.0f - params.m_particleRadius;
	if (nPos.z < -20.0f + params.m_particleRadius)
		nPos.z = -20.0f + params.m_particleRadius;

	predictedPos[index] = { nPos.x, nPos.y, nPos.z, readPos.w };
}

__global__
void calcLagrangeMultiplier(
	float4 *predictedPos,
	float4 *velocity,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells)
{
	// calculate current particle's density and lagrange multiplier.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	int3 gridPos = calcGridPosKernel(curPos);

	float density = 0.0f;
	float gradSquaredSum_j = 0.0f;
	float gradSquaredSumTotal = 0.0f;
	float3 curGrad, gradSum_i = { 0.0f,0.0f,0.0f };
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				// empty cell.
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbour = predictedPos[i];
					float3 r = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
					density += wPoly6(r);
					curGrad = wSpikyGrad(r);
					curGrad.x *= params.m_invRestDensity;
					curGrad.y *= params.m_invRestDensity;
					curGrad.z *= params.m_invRestDensity;

					gradSum_i.x += curGrad.x;
					gradSum_i.y += curGrad.y;
					gradSum_i.z += curGrad.z;
					if (i != index)
						gradSquaredSum_j += (curGrad.x * curGrad.x + curGrad.y * curGrad.y
							+ curGrad.z * curGrad.z);
				}
			}
		}
	}
	gradSquaredSumTotal = gradSquaredSum_j + gradSum_i.x * gradSum_i.x + gradSum_i.y * gradSum_i.y
		+ gradSum_i.z * gradSum_i.z;
	
	// density constraint.
	float constraint = density * params.m_invRestDensity - 1.0f;
	float lambda = -(constraint) / (gradSquaredSumTotal + params.m_lambdaEps);
	velocity[index] = { readVel.x, readVel.y, readVel.z, lambda };
}

__global__
void calcDeltaPosition(
	float4 *predictedPos,
	float4 *velocity,
	float3 *deltaPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	int3 gridPos = calcGridPosKernel(curPos);

	float curLambda = readVel.w;
	float3 deltaP = { 0.0f, 0.0f, 0.0f };
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbour = predictedPos[i];
					float neighbourLambda = velocity[i].w;
					float3 r = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
					float corrTerm = wPoly6(r) * params.m_oneDivWPoly6;
					float coff = curLambda + neighbourLambda - 0.1f * corrTerm * corrTerm * corrTerm * corrTerm;
					float3 grad = wSpikyGrad(r);
					deltaP.x += coff * grad.x;
					deltaP.y += coff * grad.y;
					deltaP.z += coff * grad.z;
				}
			}
		}
	}

	float3 ret = {deltaP.x * params.m_invRestDensity, deltaP.y * params.m_invRestDensity,
		deltaP.z * params.m_invRestDensity };
	deltaPos[index] = ret;
}

__global__
void addDeltaPosition(
	float4 *predictedPos,
	float3 *deltaPos,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readPos = predictedPos[index];
	float3 readDeltaPos = deltaPos[index];
	readDeltaPos.x = readPos.x + readDeltaPos.x;
	readDeltaPos.y = readPos.y + readDeltaPos.y;
	readDeltaPos.z = readPos.z + readDeltaPos.z;

	predictedPos[index] = { readDeltaPos.x, readDeltaPos.y, readDeltaPos.z, readPos.w };
}

__global__
void updateVelocityAndPosition(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float invDeltaTime,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	float4 oldPos = position[index];
	float4 newPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 posDiff = { newPos.x - oldPos.x, newPos.y - oldPos.y, newPos.z - oldPos.z };
	posDiff.x *= invDeltaTime;
	posDiff.y *= invDeltaTime;
	posDiff.z *= invDeltaTime;
	velocity[index] = { posDiff.x, posDiff.y, posDiff.z, readVel.w };
	position[index] = { newPos.x, newPos.y, newPos.z, newPos.w };
}

__global__
void calcVorticityConfinment(
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	float3 curVel = { readVel.x, readVel.y, readVel.z };
	int3 gridPos = calcGridPosKernel(curPos);
	float3 velDiff;
	float3 gradient;
	float3 omega = { 0.0f,0.0f,0.0f };
	float3 eta = { 0.0f,0.0f,0.0f };
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbourPos = predictedPos[i];
					float4 neighbourVel = velocity[i];
					velDiff = { neighbourVel.x - curVel.x, neighbourVel.y - curVel.y, neighbourVel.z - curVel.z };
					gradient = wSpikyGrad({ curPos.x - neighbourPos.x, curPos.y - neighbourPos.y ,
						curPos.z - neighbourPos.z });
					float3 f = cross(velDiff, gradient);
					omega.x += f.x;
					omega.y += f.y;
					omega.z += f.z;
					eta.x += gradient.x;
					eta.y += gradient.y;
					eta.z += gradient.z;
				}
			}
		}
	}
	float omegaLength = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
	float3 force = { 0.0f,0.0f,0.0f };
	//No direction for eta
	if (omegaLength == 0.0f)
	{
		deltaPos[index] = force;
		return;
	}

	// eta.
	eta.x *= omegaLength;
	eta.y *= omegaLength;
	eta.z *= omegaLength;
	if (eta.x == 0 && eta.y == 0 && eta.z == 0)
	{
		deltaPos[index] = force;
		return;
	}

	// eta normalize.
	float etaLength = sqrtf(eta.x * eta.x + eta.y * eta.y + eta.z * eta.z);
	eta.x /= etaLength;
	eta.y /= etaLength;
	eta.z /= etaLength;
	force = cross(eta, omega);
	force.x *= params.m_vorticity;
	force.y *= params.m_vorticity;
	force.z *= params.m_vorticity;
	deltaPos[index] = force;
}

__global__
void addVorticityConfinment(
	float4 *velocity,
	float3 *deltaPos,
	float deltaTime,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readVel = velocity[index];
	float3 vorticity = deltaPos[index];
	readVel.x += vorticity.x /** deltaTime*/;
	readVel.y += vorticity.y /** deltaTime*/;
	readVel.z += vorticity.z /** deltaTime*/;
	velocity[index] = readVel;
}

__global__
void calcXSPHViscosity(
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	float3 curVel = { readVel.x, readVel.y, readVel.z };
	int3 gridPos = calcGridPosKernel(curPos);
	float3 viscosity = { 0.0f,0.0f,0.0f };
	float3 velocityDiff;
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y,gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbourPos = predictedPos[i];
					float4 neighbourVel = velocity[i];
					float wPoly = wPoly6({
						curPos.x - neighbourPos.x,
						curPos.y - neighbourPos.y,
						curPos.z - neighbourPos.z });
					velocityDiff = {
						curVel.x - neighbourVel.x,
						curVel.y - neighbourVel.y,
						curVel.z - neighbourVel.z
					};
					viscosity.x += velocityDiff.x * wPoly;
					viscosity.y += velocityDiff.y * wPoly;
					viscosity.z += velocityDiff.z * wPoly;
				}
			}
		}
	}
	viscosity.x *= params.m_viscosity;
	viscosity.y *= params.m_viscosity;
	viscosity.z *= params.m_viscosity;
	deltaPos[index] = viscosity;
}

__global__
void addXSPHViscosity(
	float4 *velocity,
	float3 *deltaPos,
	float deltaTime,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readVel = velocity[index];
	float3 viscosity = deltaPos[index];
	readVel.x += viscosity.x /** deltaTime*/;
	readVel.y += viscosity.y /** deltaTime*/;
	readVel.z += viscosity.z /** deltaTime*/;
	velocity[index] = readVel;
}

#endif