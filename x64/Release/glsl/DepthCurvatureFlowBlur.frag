#version 330 core

in vec2 Texcoord;

uniform float step;
uniform sampler2D image;
uniform mat4 projectMatrix;

void main(){
	
	// -----------------reconstruct normal----------------------------
	float depth = texture(image, Texcoord).r;
	if(depth >= 1.0f || depth <= -1.0f)
	{
		gl_FragDepth = depth;
		return;
	}

	vec2 imageDim = textureSize(image, 0);
	vec2 texelSize = 1.0 / imageDim;
	
	// central differences.
	float depthRight = texture(image, Texcoord + vec2(texelSize.x, 0)).r;
	float depthLeft = texture(image, Texcoord - vec2(texelSize.x, 0)).r;
	float zdx = 0.5f * (depthRight - depthLeft);
	if(depthRight == 0.0f || depthLeft == 0.0f)
		zdx = 0.0f;
	
	float depthUp = texture(image, Texcoord + vec2(0, texelSize.y)).r;
	float depthDown = texture(image, Texcoord - vec2(0, texelSize.y)).r;
	float zdy = 0.5f * (depthUp - depthDown);
	if(depthUp == 0.0f || depthDown == 0.0f)
		zdy = 0.0f;

	float zdxx = depthRight + depthLeft - 2.0f * depth;
	float zdyy = depthUp + depthDown - 2.0f * depth;

	float depthFalloff = 0.00005f;
/*	if(abs(depth - depthRight) > depthFalloff || abs(depth - depthLeft) > depthFalloff)
		zdx = zdxx = 0.0f;
	if(abs(depth - depthDown) > depthFalloff || abs(depth - depthUp) > depthFalloff)
		zdy = zdyy = 0.0f;*/

	float Fx = projectMatrix[0][0];
	float Fy = projectMatrix[1][1];
	
	float Cx = -2.0f/(imageDim.x * Fx);
	float Cy = -2.0f/(imageDim.y * Fy);

	float D = Cy * Cy * zdx * zdx + Cx * Cx * zdy * zdy + Cx * Cx * Cy * Cy * depth;
	
	float Ex = 0.5f * zdx * dFdx(D) - zdxx * D;
	float Ey = 0.5f * zdy * dFdy(D) - zdyy * D;

	// curvature flow.
	float curvature = 0.5f * (Cy * Ex + Cx * Ey)/ pow(D, 1.5);
	if(curvature > 1.0f)
		curvature = 1.0f;

	gl_FragDepth = depth + curvature * step;
}