#version 430 core
layout (location = 0) in vec4 position;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectMatrix;
uniform float pointScale;
uniform float pointSize;
uniform float densityLowerBound;

out vec3 eyeSpacePos;

void main(){
	eyeSpacePos = (viewMatrix * modelMatrix * vec4(position.xyz, 1.0f)).xyz;
	gl_PointSize = -pointScale * pointSize / eyeSpacePos.z;
	if(position.w < densityLowerBound) 
		gl_PointSize = 0.0f;
	gl_Position = projectMatrix * viewMatrix * modelMatrix * vec4(position.xyz, 1.0f);
}