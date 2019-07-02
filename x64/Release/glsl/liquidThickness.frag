#version 430 core

uniform float pointSize;
uniform mat4 projectMatrix;

layout(location = 0) out vec4 fragColor;

void main(){	
	vec3 normal;
	normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0,1.0);
	float mag = dot(normal.xy, normal.xy);
	if(mag > 1.0) discard;
	normal.z = sqrt(1.0 - mag);
	fragColor = vec4(normal.z*0.005, 0.0, 0.0, 1.0);
}