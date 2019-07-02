#version 330 core

in vec2 Texcoord;

uniform vec2 blurDir;
uniform sampler2D image;
uniform float filterRadius;

//const float blurScale = 0.12f;
const float blurScale = 0.1f;
const float blurDepthFalloff = 100.0f;
//const float blurDepthFalloff = 0.5f;

void main(){
	// gets size of single texel.
    vec2 tex_offset = 1.0 / textureSize(image, 0);
    
	float sum = 0.0f;
	float wsum = 0.0f;
	float value = texture(image, Texcoord).r;
	
	if(value >= 1.0f || value <= -1.0f)
	{
		gl_FragDepth = value;
		return;
	}

	vec2 blurDirection = blurDir * tex_offset;
	for(float x = -filterRadius;x <= filterRadius;x += 1.0f)
	{
		float sample = texture(image, Texcoord + x * blurDirection).r;
		if(sample >= 1.0f) continue;
		
		float r = x * blurScale;
		float w = exp(-r * r);

		float r2 = (sample - value) * blurDepthFalloff;
		float g = exp(-r2 * r2);
		
		sum += sample * w * g;
		wsum += w * g;
	}
	
	if(wsum >= 0.0f)
		sum /= wsum;

	gl_FragDepth = sum;
}