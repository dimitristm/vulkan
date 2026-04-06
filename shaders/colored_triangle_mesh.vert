#version 450
#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec3 position;
layout(location = 1) in float u;
layout(location = 2) in vec3 normal;
layout(location = 3) in float v;
layout(location = 4) in vec4 in_color;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec2 out_uv;

layout(push_constant) uniform constants {
    mat4 matrix;
} view_proj_matrix;

void main()
{
	gl_Position = view_proj_matrix.matrix * vec4(position, 1.0f);
	out_color = in_color;
	out_uv.x = u;
	out_uv.y = v;
}
