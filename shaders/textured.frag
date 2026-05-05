#version 450

//shader input
layout (location = 0) in vec4 inColor;
layout (location = 1) in vec2 inUV;

//output write
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;

void main()
{
	outFragColor = texture(sampler2D(tex, samp), inUV);
}
