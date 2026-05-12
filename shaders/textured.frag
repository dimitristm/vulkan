#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;
layout(location = 2) flat in int texture_index;

layout(location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform texture2D textures[1000];
layout(set = 0, binding = 1) uniform sampler samp;

void main()
{
    outFragColor = texture(
        sampler2D( textures[nonuniformEXT(texture_index)], samp),
        in_uv
    );
}
