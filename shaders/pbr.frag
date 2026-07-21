#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

const float PI = 3.14159265359;

struct Material {
    uint albedo_idx;
    uint normal_map_idx;
    uint metallic_roughness_idx;
    uint _padding;
};

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec4 in_tangent;
layout(location = 2) in vec2 in_uv;
layout(location = 3) flat in Material in_material;
layout(location = 7) in vec3 in_frag_pos;

layout(location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform texture2D textures[1000];
layout(set = 0, binding = 1) uniform sampler samp;

layout(scalar, push_constant) uniform constants {
    mat4 view_proj_matrix;
    vec3 camera_pos;
    vec3 frag_to_light_dir;
    vec3 light_color;
} pc;

vec3 tangent_space_normal_from_RG_pixel(vec2 normal_RG) {
    vec2 xy = normal_RG * 2.0 - 1.0;
    float z = sqrt(max(1.0 - dot(xy, xy), 0.0));// dot(xy, xy) is the squared length of xy. z=sqrt(1-len(xy)^2) derived from unit sphere equation
    return vec3(xy, z);
}

vec3 world_space_normal(vec3 geometry_normal, vec4 geometry_tangent, vec2 normal_RG) {
    vec3 N = normalize(geometry_normal);
    vec3 T = normalize(geometry_tangent.xyz);

    // Gram-Schmidt orthogonalization fixes T if it's not completely orthogonal to N
    // which might happen because of interpolations from vert in to frag out vars
    T = normalize(T - dot(T, N) * N);

    // Reconstruct bitangent using the handedness sign stored in geometry_tangent.w
    vec3 B = cross(N, T) * geometry_tangent.w;
    mat3 TBN = mat3(T, B, N);

    vec3 tangent_space_normal = tangent_space_normal_from_RG_pixel(normal_RG);
    return normalize(TBN * tangent_space_normal);
}

vec3 burley_diffuse_BRDF(vec3 albedo, float roughness, float NdotL, float NdotV, float LdotH) {
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

vec3 fresnel_schlick(float VdotH, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

vec3 specular_BRDF(float NdotH, float NdotL, float NdotV, float VdotH, float roughness, vec3 F) {
    float D = NDF_GGX(NdotH, roughness);
    float Vis = visibility_Smith_GGX_correlated(NdotL, NdotV, roughness);
    return D * Vis * F;
}

// N = Normal, V = view vec (from frag to camera), L = light (from frag to light). All in world space
vec3 BRDF(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic) {
    vec3 H = normalize(V + L);

    // Dot Products
    float NdotL = max(dot(N, L), 0.0001);
    float NdotV = max(dot(N, V), 0.0001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float LdotH = max(dot(L, H), 0.0);

    // BRDF Evaluation
    vec3 diffuse = burley_diffuse_BRDF(albedo, roughness, NdotL, NdotV, LdotH);

    vec3 base_reflectivity = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnel_schlick(VdotH, base_reflectivity);

    vec3 specular = specular_BRDF(NdotH, NdotL, NdotV, VdotH, roughness, F);

    // Energy Conservation & Integration
    vec3 diffuse_factor_kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 out_color = (diffuse_factor_kD * diffuse + specular) * pc.light_color * NdotL;
    return out_color;
}

void main() {
    vec4 tex_color = texture(sampler2D(textures[nonuniformEXT(in_material.albedo_idx)], samp), in_uv);
    vec3 albedo = tex_color.rgb;
    float alpha = tex_color.a;

    vec2 normal_RG = texture(sampler2D(textures[nonuniformEXT(in_material.normal_map_idx)], samp), in_uv).rg;
    vec2 rough_metal_RG = texture(sampler2D(textures[nonuniformEXT(in_material.metallic_roughness_idx)], samp), in_uv).rg;

    float roughness = clamp(rough_metal_RG.r, 0.04, 1.0);
    float metallic = rough_metal_RG.g;

    vec3 N = world_space_normal(in_normal, in_tangent, normal_RG);
    vec3 V = normalize(pc.camera_pos - in_frag_pos);
    vec3 L = normalize(pc.frag_to_light_dir);

    vec3 out_color = BRDF(N, V, L, albedo, roughness, metallic);

    outFragColor = vec4(out_color, alpha);
}
