#version 450
#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec3 position;
layout(location = 1) in float u;
layout(location = 2) in vec3 normal;
layout(location = 3) in float v;
layout(location = 4) in vec3 in_color;

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uv;

void main()
{
	gl_Position = vec4(position, 1.0f);
	out_color = in_color.xyz;
	out_uv.x = u;
	out_uv.y = v;
}
