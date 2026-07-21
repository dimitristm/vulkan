#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

struct Material {
    uint albedo_idx;
    uint normal_map_idx;
    uint metallic_roughness_idx;
    uint _padding;
};

struct InstanceInfo {
    mat4 model_transform;
    Material material;
};

layout(location = 0) in vec3 position;
layout(location = 1) in float u;
layout(location = 2) in vec3 normal;
layout(location = 3) in float v;
layout(location = 4) in vec4 tangent;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec4 out_tangent;
layout(location = 2) out vec2 out_uv;
layout(location = 3) flat out Material out_material;
layout(location = 7) out vec3 out_frag_pos;

layout(std430, set = 0, binding = 2) readonly buffer InstanceInfoBuffer {
    InstanceInfo instance_infos[];
} instance_info_buffer;

layout(scalar, push_constant) uniform constants {
    mat4 view_proj_matrix;
    vec3 camera_pos;
    vec3 light_dir;
    vec3 light_color;
} pc;

void main() {
    InstanceInfo instance_info = instance_info_buffer.instance_infos[gl_InstanceIndex];
    vec4 world_pos = instance_info.model_transform * vec4(position, 1.0);

    mat3 normal_matrix = mat3(transpose(inverse(instance_info.model_transform)));

    gl_Position = pc.view_proj_matrix * world_pos;

    out_normal = normal_matrix * normal;
    out_tangent = vec4(normal_matrix * tangent.xyz, tangent.w);
    out_uv = vec2(u, v);
    out_material = instance_info.material;
    out_frag_pos = world_pos.xyz;
}
