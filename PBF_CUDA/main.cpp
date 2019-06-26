#include <string>
#include <iostream>

#include "Renderer/RenderDevice.h"
#include "Renderer/Camera/TPSCamera.h"
#include "Renderer/Manager/Geometry.h"
#include "Renderer/Drawable/InstanceDrawable.h"
#include "Renderer/Drawable/StaticModelDrawable.h"
#include "Renderer/Drawable/ParticleInstanceDrawable.h"
#include "Renderer/Drawable/ParticlePointSpriteDrawable.h"
#include "Renderer/Voxelization.h"
#include "Simulator/FluidDemo.h"

using namespace std;
using namespace Renderer;
using namespace Simulator;

int main(int argc, char *argv[])
{
	// initialization.
	const int width = 1280, height = 780;
	RenderDevice::ptr window = RenderDevice::getSingleton();
	window->initialize("Fluid Simulation", width, height, false);
	RenderSystem::ptr renderSys = window->getRenderSystem();

	// resource loading.
	MeshMgr::ptr meshMgr = renderSys->getMeshMgr();
	ShaderMgr::ptr shaderMgr = renderSys->getShaderMgr();
	TextureMgr::ptr textureMgr = renderSys->getTextureMgr();
	// shaders.
	unsigned int simpleColorShader = shaderMgr->loadShader("simple_color",
		"./glsl/simple_color.vert", "./glsl/simple_color.frag");
	unsigned int simpleTextShader = shaderMgr->loadShader("simple_texture",
		"./glsl/simple_texture.vert", "./glsl/simple_texture.frag");
	unsigned int blinnPhongShader = shaderMgr->loadShader("blinn_phong",
		"./glsl/blinn_phong.vert", "./glsl/blinn_phong.frag");
	// textures.
	unsigned int containerTex = textureMgr->loadTexture2D("cube", "./res/floor.png");
	unsigned int fluidTex = textureMgr->loadTexture2D("blue", "./res/blue.png");
	renderSys->setSkyDome("./res/skybox/", ".jpg");
	// meshes.
	unsigned int planeMesh = meshMgr->loadMesh(new Plane(1.0, 1.0));
	unsigned int containerMesh = meshMgr->loadMesh(new Container(1.0f, 1.0f, 1.0f));
	unsigned int particleSphereMesh = meshMgr->loadMesh(new Sphere(1.0f, 5, 5));
	unsigned int particleCubeMesh = meshMgr->loadMesh(new Cube(1.0f, 1.0f, 1.0f));
	// sunlight.
	renderSys->setSunLight(glm::vec3(1.0f, 0.5f, 0.5f),
		glm::vec3(0.05f), glm::vec3(0.6f), glm::vec3(0.6f));
	// shadow setting.
	renderSys->createShadowDepthBuffer(4096, 4096);
	renderSys->createSunLightCamera(glm::vec3(0.0f), -600.0f, +600.0f,
		-600.0f, +600.0f, 1.0f, 500.0f);

	StaticModelDrawable *island = new StaticModelDrawable(blinnPhongShader,
		"./res/Small Tropical Island/Small Tropical Island.obj");
	renderSys->addDrawable(island);

	glm::vec3 scaleSize = glm::vec3(80.0f, 40.0f, 40.0f);
	glm::vec3 center = glm::vec3(0, scaleSize.y/2 + 2.0f, 150.0f);
	SimpleDrawable *container[6];
	container[0] = new SimpleDrawable(blinnPhongShader);
	container[1] = new SimpleDrawable(blinnPhongShader);
	container[2] = new SimpleDrawable(blinnPhongShader);
	container[3] = new SimpleDrawable(blinnPhongShader);
	container[4] = new SimpleDrawable(blinnPhongShader);
	container[5] = new SimpleDrawable(blinnPhongShader);
	container[0]->addMesh(planeMesh);
	container[1]->addMesh(planeMesh);
	container[2]->addMesh(planeMesh);
	container[3]->addMesh(planeMesh);
	container[4]->addMesh(planeMesh);
	container[5]->addMesh(planeMesh);
	container[0]->addTexture(containerTex);
	container[1]->addTexture(containerTex);
	container[2]->addTexture(containerTex);
	container[3]->addTexture(containerTex);
	container[4]->addTexture(containerTex);
	container[5]->addTexture(containerTex);
	container[0]->setReceiveShadow(false);
	container[1]->setReceiveShadow(false);
	container[2]->setReceiveShadow(false);
	container[3]->setReceiveShadow(false);
	container[4]->setReceiveShadow(false);
	container[5]->setReceiveShadow(false);
	container[0]->setProduceShadow(true);
	container[1]->setProduceShadow(true);
	container[2]->setProduceShadow(true);
	container[3]->setProduceShadow(true);
	container[4]->setProduceShadow(true);
	container[5]->setProduceShadow(true);
	container[0]->getTransformation()->setScale(glm::vec3(scaleSize.y, scaleSize.y, scaleSize.z));
	container[1]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[2]->getTransformation()->setScale(glm::vec3(scaleSize.y, scaleSize.y, scaleSize.z));
	container[3]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[4]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[5]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[0]->getTransformation()->rotate(glm::vec3(0, 0, 1), 270.0f);
	container[1]->getTransformation()->rotate(glm::vec3(0, 0, 1), 180.0f);
	container[2]->getTransformation()->rotate(glm::vec3(0, 0, 1), 90.00f);
	container[4]->getTransformation()->rotate(glm::vec3(1, 0, 0), 270.0f);
	container[5]->getTransformation()->rotate(glm::vec3(1, 0, 0), 90.00f);
	container[0]->getTransformation()->setTranslation(glm::vec3(center.x - scaleSize.x/2.0f, center.y, center.z));
	container[1]->getTransformation()->setTranslation(glm::vec3(center.x, center.y + scaleSize.y / 2.0f, center.z));
	container[2]->getTransformation()->setTranslation(glm::vec3(center.x + scaleSize.x / 2.0f, center.y, center.z));
	container[3]->getTransformation()->setTranslation(glm::vec3(center.x, center.y - scaleSize.y / 2.0f, center.z));
	container[4]->getTransformation()->setTranslation(glm::vec3(center.x , center.y, center.z + scaleSize.z / 2.0f));
	container[5]->getTransformation()->setTranslation(glm::vec3(center.x, center.y, center.z - scaleSize.z / 2.0f));
	renderSys->addDrawable(container[0]);
	renderSys->addDrawable(container[1]);
	renderSys->addDrawable(container[2]);
	renderSys->addDrawable(container[3]);
	renderSys->addDrawable(container[4]);
	renderSys->addDrawable(container[5]);

	Camera3D::ptr camera = renderSys->createTPSCamera(
		glm::vec3(0, 0, 0),
		glm::vec3(center.x, center.y, center.z));
	camera->setPerspectiveProject(45.0f, static_cast<float>(width) / height, 0.1f, 3000.0f);
	TPSCamera *tpsCamera = reinterpret_cast<TPSCamera*>(camera.get());
	tpsCamera->setPitch(15.0f);
	tpsCamera->setDistance(90.0f);
	tpsCamera->setDistanceLimt(0.01f, 1000.0f);
	tpsCamera->setWheelSensitivity(5.0f);
	FluidDemo demo;
	//demo.boxFluidDrop(65536, 128, scaleSize.x, 0.3f);
	//demo.boxFluidDrop(65536, 128, scaleSize.x, 0.35f);
	demo.boxFluidDrop(128000, 128, scaleSize.x, 0.32f);
	//demo.boxFluidDrop(256000, 128, scaleSize.x, 0.27f);
	//demo.boxFluidDrop(256000, 128, scaleSize.x, 0.15f);
	//demo.boxFluidDrop(512000, 128, scaleSize.x, 0.25f);
	FluidSystem::ptr fluid = demo.getSimulator();
	ParticleInstanceDrawable *particleDrawable = new ParticleInstanceDrawable(3);
	particleDrawable->addMesh(particleSphereMesh);
	//particleDrawable->addMesh(particleCubeMesh);
	particleDrawable->addTexture(fluidTex);
	particleDrawable->setProduceShadow(false);
	particleDrawable->setParticleRadius(fluid->getSimulateParams().m_particleRadius);
	particleDrawable->getTransformation()->setTranslation(center);
	particleDrawable->setInstanceVBO(fluid->getPosVBO(), fluid->getSimulateParams().m_numParticles);
	renderSys->addDrawable(particleDrawable);

	renderSys->setGlowBlur(true);
	bool run = false;
	bool reset = false;
	bool wallMove = false;
	float wallMoveSpeed = +10.0f;
	float curWallPos = -40.0f;
	while (window->run())
	{
		// wall movement.
		if (wallMove)
		{
			if (curWallPos >= -15.0f)
			{
				curWallPos = -15.0f;
				wallMoveSpeed = -wallMoveSpeed;
			}
			if (curWallPos < -40.0f)
			{
				curWallPos = -40.0f;
				wallMoveSpeed = -wallMoveSpeed;
			}
			curWallPos += wallMoveSpeed * window->m_deltaTime;
			fluid->getSimulateParams().m_leftWall = curWallPos;
			container[0]->getTransformation()->setTranslation(glm::vec3(curWallPos, center.y, center.z));
		}

		// simulation.
		//if(run)
		//	fluid->simulate(0.0083f);
		if (run)
			fluid->simulate(0.012f);

		window->beginFrame();

		renderSys->render();

		// Demo setting window.
		{
			ImGui::Begin("Fluid simulation");
			ImGui::SliderFloat("Viscosity", &fluid->getSimulateParams().m_viscosity, 0.0f, 1.0f);
			ImGui::SliderFloat("Vorticity", &fluid->getSimulateParams().m_vorticity, 0.0f, 0.1f);
			if (ImGui::Button("Wall Movement"))
				wallMove = !wallMove;
			if (ImGui::Button("Reset"))
				reset = true;
			if (ImGui::Button("Run"))
				run = !run;
			ImGui::End();
		}

		if (reset)
		{
			reset = false;
			demo.boxFluidDrop(65536, 128, scaleSize.x, 0.3f);
			//demo.boxFluidDrop(35562, 128, scaleSize.x, 0.3f);
			fluid = demo.getSimulator();
			particleDrawable->setInstanceVBO(fluid->getPosVBO(), fluid->getSimulateParams().m_numParticles);
		}

		window->endFrame();
	}

	fluid->finalize();
	window->shutdown();

	return 0;
}