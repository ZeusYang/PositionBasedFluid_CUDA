#version 430 core

uniform float pointSize;
uniform mat4 projectMatrix;

in vec3 eyeSpacePos;

void main(){	
	vec3 normal;
	normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0,1.0);
	float mag = dot(normal.xy, normal.xy);
	if(mag > 1.0) discard;
	normal.z = sqrt(1.0 - mag);

	vec4 pixelEyePos = vec4(eyeSpacePos + normal * pointSize, 1.0f);
	vec4 pixelClipPos = projectMatrix * pixelEyePos;
	float ndcZ = pixelClipPos.z / pixelClipPos.w;
	gl_FragDepth = ndcZ;
}