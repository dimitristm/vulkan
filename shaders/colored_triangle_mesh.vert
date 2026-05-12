#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 position;
layout(location = 1) in float u;
layout(location = 2) in vec3 normal;
layout(location = 3) in float v;
layout(location = 4) in vec4 in_color;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;
layout(location = 2) flat out int texture_index;

layout(push_constant) uniform constants {
    mat4 view_proj_matrix;
} pc;

layout(set = 0, binding = 2) readonly buffer ObjectTransformIndices {
    int object_transform_indices[];
};

layout(set = 0, binding = 3) readonly buffer AlbedoTextureIndices {
    int albedo_texture_indices[];
};

layout(set = 0, binding = 4) readonly buffer ObjectTransforms {
    mat4 object_transforms[];
};

void main()
{
    int object_transform_index = object_transform_indices[gl_InstanceIndex];
    texture_index = albedo_texture_indices[gl_InstanceIndex];
    mat4 model = object_transforms[object_transform_index];

    gl_Position = pc.view_proj_matrix * model * vec4(position, 1.0);

    out_color = in_color;
    out_uv = vec2(u, v);
}
