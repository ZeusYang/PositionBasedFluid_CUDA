#include <cuda_runtime.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "thrust/tuple.h"
#include "thrust/functional.h"

#include "FluidKernel.cuh"

void getLastCudaError(const char * errorMessage)
{
	// check cuda last error.
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::cout << "getLastCudaError() CUDA error : "
			<< errorMessage << " : " << "(" << static_cast<int>(err) << ") "
			<< cudaGetErrorString(err) << ".\n";
	}
}

void setParameters(SimulateParams *hostParams)
{
	cudaMemcpyToSymbol(params, hostParams, sizeof(SimulateParams));
}

void computeHash(
	unsigned int *gridParticleHash,
	float *pos,
	int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);

	// launch the kernel.
	calcParticlesHashKernel << <numBlocks, numThreads >> > (
		gridParticleHash,
		(float4*)pos,
		numParticles);
}

void sortParticles(
	unsigned int *deviceGridParticleHash,
	unsigned int numParticles,
	float *devicePos,
	float *deviceVel,
	float *devicePredictedPos
)
{
	thrust::device_ptr<float4> ptrPos((float4*)devicePos);
	thrust::device_ptr<float4> ptrVel((float4*)deviceVel);
	thrust::device_ptr<float4> ptrPredictedPos((float4*)devicePredictedPos);
	thrust::sort_by_key(
		thrust::device_ptr<unsigned int>(deviceGridParticleHash),
		thrust::device_ptr<unsigned int>(deviceGridParticleHash + numParticles),
		thrust::make_zip_iterator(thrust::make_tuple(ptrPos, ptrVel, ptrPredictedPos)));
}

void findCellRange(
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCell)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	
	// set all cell to empty.
	cudaMemset(cellStart, 0xffffffff, numCell * sizeof(unsigned int));

	unsigned int memSize = sizeof(unsigned int) * (numThreads + 1);
	findCellRangeKernel << <numBlocks, numThreads, memSize >> > (
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
}

void fluidAdvection(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	
	advect << < numBlocks, numThreads >> > (
		position,
		velocity,
		predictedPos,
		deltaTime,
		numParticles);
	cudaDeviceSynchronize();
}

void densityConstraint(
	float4 *position,
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	
	// calculate density and lagrange multiplier.
	calcLagrangeMultiplier << <numBlocks, numThreads >> > (
		predictedPos,
		velocity,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles,
		numCells);
	getLastCudaError("calcLagrangeMultiplier");
	//cudaDeviceSynchronize();
	
	// calculate delta position.
	calcDeltaPosition << <numBlocks, numThreads >> > (
		predictedPos,
		velocity,
		deltaPos,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
	getLastCudaError("calcDeltaPosition");
	//cudaDeviceSynchronize();

	// add delta position.
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	addDeltaPosition << <numBlocks, numThreads >> > (
		predictedPos,
		deltaPos,
		numParticles);
	getLastCudaError("addDeltaPosition");
	//cudaDeviceSynchronize();
}

void updateVelAndPos(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float deltaTime,
	unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	updateVelocityAndPosition << <numBlocks, numThreads >> > (
		position,
		velocity,
		predictedPos,
		1.0f / deltaTime,
		numParticles);
	//cudaDeviceSynchronize();
}
void applyVorticityConfinment(
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	float deltaTime,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	cudaDeviceSynchronize();
	calcVorticityCurl << <numBlocks, numThreads >> > (
		velocity,
		deltaPos,
		predictedPos,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
	cudaDeviceSynchronize();

	calcVorticityEta << <numBlocks, numThreads >> > (
		velocity,
		deltaPos,
		predictedPos,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
	cudaDeviceSynchronize();
	//addVorticityConfinment << <numBlocks, numThreads >> > (
	//	velocity,
	//	deltaPos,
	//	deltaTime,
	//	numParticles);
}

void applyXSPHViscosity(
	float4 *velocity,
	float3 *deltaPos,
	float4 *predictedPos,
	float deltaTime,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	unsigned int numThreads, numBlocks;
	numThreads = 256;
	numBlocks = (numParticles % numThreads != 0) ? (numParticles / numThreads + 1) : (numParticles / numThreads);
	calcXSPHViscosity << <numBlocks, numThreads >> > (
		velocity,
		deltaPos,
		predictedPos,
		cellStart,
		cellEnd,
		gridParticleHash,
		numParticles);
	cudaDeviceSynchronize();

	addXSPHViscosity << <numBlocks, numThreads >> > (
		velocity,
		deltaPos,
		deltaTime,
		numParticles);
}