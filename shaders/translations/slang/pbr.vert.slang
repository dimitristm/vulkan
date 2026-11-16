struct Material {
    uint albedo_idx;
    uint normal_map_idx;
    uint metallic_roughness_idx;
    uint _padding;
};

struct InstanceInfo {
    column_major float4x4 model_transform;
    Material material;
};

[[vk::binding(2, 0)]] StructuredBuffer<InstanceInfo> instance_infos;

struct Constants {
    [[vk::offset(0)]] column_major float4x4 view_proj_matrix;
    [[vk::offset(64)]] float3 camera_pos;
    [[vk::offset(76)]] float3 light_dir;
    [[vk::offset(88)]] float3 light_color;
};

[[vk::push_constant]] ConstantBuffer<Constants> pc;

struct VertexInput {
    [[vk::location(0)]] float3 position : POSITION0;
    [[vk::location(1)]] float u : TEXCOORD0;
    [[vk::location(2)]] float3 normal : NORMAL0;
    [[vk::location(3)]] float v : TEXCOORD1;
    [[vk::location(4)]] float4 tangent : TANGENT0;
};

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : NORMAL0;
    [[vk::location(1)]] float4 tangent : TANGENT0;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] nointerpolation uint albedo_idx : TEXCOORD1;
    [[vk::location(4)]] nointerpolation uint normal_map_idx : TEXCOORD2;
    [[vk::location(5)]] nointerpolation uint metallic_roughness_idx : TEXCOORD3;
    [[vk::location(6)]] nointerpolation uint material_padding : TEXCOORD4;
    [[vk::location(7)]] float3 frag_pos : TEXCOORD5;
};

float4x4 inverse_transpose(float4x4 m) {
    float4x4 cofactors;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float3x3 minor_matrix;
            int minor_row = 0;
            for (int r = 0; r < 4; ++r) {
                if (r == row) continue;
                int minor_col = 0;
                for (int c = 0; c < 4; ++c) {
                    if (c == col) continue;
                    minor_matrix[minor_row][minor_col] = m[r][c];
                    ++minor_col;
                }
                ++minor_row;
            }
            float sign = ((row + col) % 2 == 0) ? 1.0 : -1.0;
            cofactors[row][col] = sign * determinant(minor_matrix);
        }
    }
    return cofactors / dot(m[0], cofactors[0]);
}

[shader("vertex")]
VertexOutput main(VertexInput input, uint instance_index : SV_InstanceID) {
    VertexOutput output;
    InstanceInfo instance_info = instance_infos[instance_index];
    float4 world_pos = mul(instance_info.model_transform, float4(input.position, 1.0));

    float3x3 normal_matrix = (float3x3)inverse_transpose(instance_info.model_transform);

    output.position = mul(pc.view_proj_matrix, world_pos);

    output.normal = mul(normal_matrix, input.normal);
    output.tangent = float4(mul(normal_matrix, input.tangent.xyz), input.tangent.w);
    output.uv = float2(input.u, input.v);
    output.albedo_idx = instance_info.material.albedo_idx;
    output.normal_map_idx = instance_info.material.normal_map_idx;
    output.metallic_roughness_idx = instance_info.material.metallic_roughness_idx;
    output.material_padding = instance_info.material._padding;
    output.frag_pos = world_pos.xyz;
    return output;
}
