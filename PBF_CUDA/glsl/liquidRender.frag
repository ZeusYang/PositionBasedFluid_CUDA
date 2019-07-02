#version 430 core

in vec2 Texcoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColor;

uniform float pointSize;
uniform mat4 viewMatrix;
uniform mat4 invViewMatrix;
uniform mat4 projectMatrix;
uniform vec4 liquidColor;

uniform sampler2D depthTex;
uniform sampler2D thicknessTex;
uniform sampler2D backgroundTex;
uniform sampler2D backgroundDepthTex;

uniform mat4 invProjectMatrix;

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform vec3 cameraPos;
uniform DirLight dirLight;

vec3 uvToEye(vec2 coord, float z)
{
	vec2 pos = coord * 2.0f - 1.0f;
	vec4 clipPos = vec4(pos, z, 1.0f);
	vec4 viewPos = invProjectMatrix * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main(){	

	float backgroundDepth = texture(backgroundDepthTex, Texcoord).r;

	float depth = texture(depthTex, Texcoord).r;
	if(depth <= -1.0f || depth >= 1.0f || (depth*0.5f + 0.5f > backgroundDepth))
	{
		fragColor = texture(backgroundTex, Texcoord);
		return;
	}

	// -----------------reconstruct normal----------------------------
	vec2 depthTexelSize = 1.0 / textureSize(depthTex, 0);
	// calculate eye space position.
	vec3 eyeSpacePos = uvToEye(Texcoord, depth);
	// finite difference.
	vec3 ddxLeft   = eyeSpacePos - uvToEye(Texcoord - vec2(depthTexelSize.x,0.0f),
					texture(depthTex, Texcoord - vec2(depthTexelSize.x,0.0f)).r);
	vec3 ddxRight  = uvToEye(Texcoord + vec2(depthTexelSize.x,0.0f),
					texture(depthTex, Texcoord + vec2(depthTexelSize.x,0.0f)).r) - eyeSpacePos;
	vec3 ddyTop    = uvToEye(Texcoord + vec2(0.0f,depthTexelSize.y),
					texture(depthTex, Texcoord + vec2(0.0f,depthTexelSize.y)).r) - eyeSpacePos;
	vec3 ddyBottom = eyeSpacePos - uvToEye(Texcoord - vec2(0.0f,depthTexelSize.y),
					texture(depthTex, Texcoord - vec2(0.0f,depthTexelSize.y)).r);
	vec3 dx = ddxLeft;
	vec3 dy = ddyTop;
	if(abs(ddxRight.z) < abs(ddxLeft.z))
		dx = ddxRight;
	if(abs(ddyBottom.z) < abs(ddyTop.z))
		dy = ddyBottom;
	vec3 normal = normalize(cross(dx, dy));
	vec3 worldPos = (invViewMatrix * vec4(eyeSpacePos, 1.0f)).xyz;

	// -----------------refracted----------------------------
	vec2 texScale = vec2(0.75, 1.0);		// ???.
	float refractScale = 1.33 * 0.025;	// index.
	refractScale *= smoothstep(0.1, 0.4, worldPos.y);
	vec2 refractCoord = Texcoord + normal.xy * refractScale * texScale;
	float thickness = max(texture(thicknessTex, Texcoord).r, 0.3f);
	vec3 transmission = exp(-(vec3(1.0f) - liquidColor.xyz) * thickness);
	vec3 refractedColor = texture(backgroundTex, refractCoord).xyz * transmission;

	// -----------------Phong lighting----------------------------
	vec3 viewDir = -normalize(eyeSpacePos);
	vec3 lightDir = normalize((viewMatrix * vec4(dirLight.direction, 0.0f)).xyz);
	vec3 halfVec = normalize(viewDir + lightDir);
	vec3 specular = vec3(dirLight.specular * pow(max(dot(halfVec, normal), 0.0f), 400.0f));
	vec3 diffuse = liquidColor.xyz * max(dot(lightDir, normal), 0.0f) * dirLight.diffuse * liquidColor.w;
	
	// -----------------Merge all effect----------------------------
	fragColor.rgb = diffuse + specular + refractedColor;
	fragColor.a = 1.0f;

	// gamma correction.
	// glow map.
	float brightness = dot(fragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
	brightColor = vec4(fragColor.rgb * brightness * brightness, 1.0f);
}