#version 330 core

in vec2 Texcoord;

uniform sampler2D image;
uniform float filterRadius;

const float blurScale = 0.05f;
const float blurDepthFalloff = 500.0f;

void main(){
	// gets size of single texel.
    vec2 tex_offset = 1.0 / textureSize(image, 0);
    
	float sum = 0.0f;
	float wsum = 0.0f;
	float value = texture(image, Texcoord).r;

	for(float y = -filterRadius;y <= filterRadius;y += 1.0f)
	{
		for(float x = -filterRadius;x <= filterRadius;x += 1.0f)
		{
			float sample = texture(image, Texcoord + vec2(x, y) * tex_offset).r;
			
			// spatial domain.
			float r = length(vec2(x, y)) * blurScale;
			float w = exp(-r * r);

			// range domain.
			float r2 = (sample - value) * blurDepthFalloff;
			float g = exp(-r2 * r2);
			
			sum += sample * w * g;
			wsum += w * g;
		}
	}

	if(wsum >= 0.0f)
		sum /= wsum;

	gl_FragDepth = sum;
}