static const float PI = 3.14159265359;

struct FragmentInput {
    [[vk::location(0)]] float3 normal : NORMAL0;
    [[vk::location(1)]] float4 tangent : TANGENT0;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] nointerpolation uint albedo_idx : TEXCOORD1;
    [[vk::location(4)]] nointerpolation uint normal_map_idx : TEXCOORD2;
    [[vk::location(5)]] nointerpolation uint metallic_roughness_idx : TEXCOORD3;
    [[vk::location(6)]] nointerpolation uint material_padding : TEXCOORD4;
    [[vk::location(7)]] float3 frag_pos : TEXCOORD5;
};

[[vk::binding(0, 0)]] Texture2D<float4> textures[1000];
[[vk::binding(1, 0)]] SamplerState samp;

struct Constants {
    [[vk::offset(0)]] column_major float4x4 view_proj_matrix;
    [[vk::offset(64)]] float3 camera_pos;
    [[vk::offset(76)]] float3 frag_to_light_dir;
    [[vk::offset(88)]] float3 light_color;
};

[[vk::push_constant]] ConstantBuffer<Constants> pc;

float3 tangent_space_normal_from_RG_pixel(float2 normal_RG) {
    float2 xy = normal_RG * 2.0 - 1.0;
    float z = sqrt(max(1.0 - dot(xy, xy), 0.0));// dot(xy, xy) is the squared length of xy. z=sqrt(1-len(xy)^2) derived from unit sphere equation
    return float3(xy, z);
}

float3 world_space_normal(float3 geometry_normal, float4 geometry_tangent, float2 normal_RG) {
    float3 N = normalize(geometry_normal);
    float3 T = normalize(geometry_tangent.xyz);

    // Gram-Schmidt orthogonalization fixes T if it's not completely orthogonal to N
    // which might happen because of interpolations from vert in to frag out vars
    T = normalize(T - dot(T, N) * N);

    // Reconstruct bitangent using the handedness sign stored in geometry_tangent.w
    float3 B = cross(N, T) * geometry_tangent.w;
    float3x3 TBN = transpose(float3x3(T, B, N));

    float3 tangent_space_normal = tangent_space_normal_from_RG_pixel(normal_RG);
    return normalize(mul(TBN, tangent_space_normal));
}

float3 burley_diffuse_BRDF(float3 albedo, float roughness, float NdotL, float NdotV, float LdotH) {
    float fd90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    float light_scatter = 1.0 + (fd90 - 1.0) * pow(clamp(1.0 - NdotL, 0.0, 1.0), 5.0);
    float view_scatter  = 1.0 + (fd90 - 1.0) * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    return (albedo / PI) * light_scatter * view_scatter;
}

float NDF_GGX(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denominator = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denominator * denominator);
}

// Combines the geometric shadowing G and BRDF denominator into one term
float visibility_Smith_GGX_correlated(float NdotL, float NdotV, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float lambdaV = NdotL * sqrt(NdotV * NdotV * (1.0 - alpha2) + alpha2);
    float lambdaL = NdotV * sqrt(NdotL * NdotL * (1.0 - alpha2) + alpha2);
    return 0.5 / max(lambdaV + lambdaL, 0.0001);
}

float3 fresnel_schlick(float VdotH, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

float3 specular_BRDF(float NdotH, float NdotL, float NdotV, float VdotH, float roughness, float3 F) {
    float D = NDF_GGX(NdotH, roughness);
    float Vis = visibility_Smith_GGX_correlated(NdotL, NdotV, roughness);
    return D * Vis * F;
}

// N = Normal, V = view vec (from frag to camera), L = light (from frag to light). All in world space
float3 BRDF(float3 N, float3 V, float3 L, float3 albedo, float roughness, float metallic) {
    float3 H = normalize(V + L);

    // Dot Products
    float NdotL = max(dot(N, L), 0.0001);
    float NdotV = max(dot(N, V), 0.0001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float LdotH = max(dot(L, H), 0.0);

    // BRDF Evaluation
    float3 diffuse = burley_diffuse_BRDF(albedo, roughness, NdotL, NdotV, LdotH);

    float3 base_reflectivity = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float3 F = fresnel_schlick(VdotH, base_reflectivity);

    float3 specular = specular_BRDF(NdotH, NdotL, NdotV, VdotH, roughness, F);

    // Energy Conservation & Integration
    float3 diffuse_factor_kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float3 out_color = (diffuse_factor_kD * diffuse + specular) * pc.light_color * NdotL;
    return out_color;
}

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0 {
    float4 tex_color = textures[NonUniformResourceIndex(input.albedo_idx)].Sample(samp, input.uv);
    float3 albedo = tex_color.rgb;
    float alpha = tex_color.a;

    float2 normal_RG = textures[NonUniformResourceIndex(input.normal_map_idx)].Sample(samp, input.uv).rg;
    float2 rough_metal_RG = textures[NonUniformResourceIndex(input.metallic_roughness_idx)].Sample(samp, input.uv).rg;

    float roughness = clamp(rough_metal_RG.r, 0.04, 1.0);
    float metallic = rough_metal_RG.g;

    float3 N = world_space_normal(input.normal, input.tangent, normal_RG);
    float3 V = normalize(pc.camera_pos - input.frag_pos);
    float3 L = normalize(pc.frag_to_light_dir);

    float3 out_color = BRDF(N, V, L, albedo, roughness, metallic);

    return float4(out_color, alpha);
}
