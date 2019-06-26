#version 430 core
layout (location = 0) in vec4 position;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectMatrix;
uniform float pointSize;

void main(){
	vec3 eyeSpacePos = (viewMatrix * modelMatrix * vec4(position.xyz, 1.0f)).xyz;
	gl_PointSize = max(1.0f, 800.0f * pointSize / (1.0f - eyeSpacePos.z));
	gl_Position = projectMatrix * viewMatrix * modelMatrix * vec4(position.xyz, 1.0f);
}