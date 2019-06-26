#include "SimulateParams.cuh"

extern "C"
{
	void getLastCudaError(const char *errorMessage);

	void setParameters(SimulateParams *hostParams);

	void computeHash(
		unsigned int *gridParticleHash,
		float *pos,
		int numParticles);

	void sortParticles(
		unsigned int *deviceGridParticleHash,
		unsigned int numParticles,
		float *devicePos,
		float *deviceVel,
		float *devicePredictedPos
	);

	void findCellRange(
		unsigned int *cellStart,
		unsigned int *cellEnd,
		unsigned int *gridParticleHash,
		unsigned int numParticles,
		unsigned int numCell);

	void fluidAdvection(
		float4 *position,
		float4 *velocity,
		float4 *predictedPos,
		float deltaTime,
		unsigned int numParticles);
	
	void densityConstraint(
		float4 *position,
		float4 *velocity,
		float3 *deltaPos,
		float4 *predictedPos,
		unsigned int *cellStart,
		unsigned int *cellEnd,
		unsigned int *gridParticleHash,
		unsigned int numParticles,
		unsigned int numCells);

	void updateVelAndPos(
		float4 *position,
		float4 *velocity,
		float4 *predictedPos,
		float deltaTime,
		unsigned int numParticles);

	void applyVorticityConfinment(
		float4 *velocity,
		float3 *deltaPos,
		float4 *predictedPos,
		float deltaTime,
		unsigned int *cellStart,
		unsigned int *cellEnd,
		unsigned int *gridParticleHash,
		unsigned int numParticles);

	void applyXSPHViscosity(
		float4 *velocity,
		float3 *deltaPos,
		float4 *predictedPos,
		float deltaTime,
		unsigned int *cellStart,
		unsigned int *cellEnd,
		unsigned int *gridParticleHash,
		unsigned int numParticles);
}