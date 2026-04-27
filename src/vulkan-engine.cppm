module;

#include <SDL3/SDL_video.h>

#include <boost/pfr.hpp>


#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


// Disable harmless VMA warnings
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wall"
#  pragma GCC diagnostic ignored "-Wextra"
#endif

#include <vk_mem_alloc.h>

#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif


#if !USE_IMPORT_STD
#include <cstdio>
#include <vector>
#include <span>
#include <array>
#include <unordered_set>
#include <optional>
#include <algorithm>
#endif

export module vulkanEngine;
import vertexBufferAttributeTypes;
#if USE_IMPORT_STD
import std;
#endif


struct APIVersionVulkan{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;

    [[nodiscard]] uint32_t to_vk_enum()const{
        return VK_MAKE_VERSION(major, minor, patch);
    }
};

export enum class MSAALevel : std::uint8_t{
    //todo: check physical device limits, they might not support one of these
    OFF = VK_SAMPLE_COUNT_1_BIT,
    X2 = VK_SAMPLE_COUNT_2_BIT,
    X4 = VK_SAMPLE_COUNT_4_BIT,
    X8 = VK_SAMPLE_COUNT_8_BIT,
};

export template <typename T>
struct PushConstant {
    static_assert(std::is_trivially_copyable_v<T>, "Push constant type must be trivially copyable");

    const uint32_t offset;
    const uint32_t size;
    const VkShaderStageFlags shader_stages;
    T data;

    PushConstant(uint32_t offset, VkShaderStageFlags shader_stages, const T& data = {})
    :
    offset(offset),
    size(sizeof(T)),
    shader_stages(shader_stages),
    data(data)
    {}
};

export class PushConstantsBuilder{
    uint32_t current_last_byte_used = 0;
    std::vector<VkPushConstantRange> ranges;

    static bool range_stages_do_not_overlap(const std::vector<VkPushConstantRange>& old, VkShaderStageFlags new_flags);

public:
    PushConstantsBuilder() = default;

    template <typename T>
    PushConstant<T> add(VkShaderStageFlags shader_stages){
        static_assert(sizeof(T) % 4 == 0, "Size of push constant must be multiple of 4");
        assert(range_stages_do_not_overlap(ranges, shader_stages));

        VkPushConstantRange range{
            .stageFlags = shader_stages,
            .offset = current_last_byte_used,
            .size = sizeof(T),
        };

        ranges.push_back(range);

        current_last_byte_used += sizeof(T);

        assert(current_last_byte_used <= 128 && "Assert failed: used more push constant space than is guaranteed to exist");

        return PushConstant<T>{range.offset, range.stageFlags};
    }

    [[nodiscard]] const std::vector<VkPushConstantRange>& get_ranges() const{ return ranges; }

    PushConstantsBuilder &reset();
};

export struct VulkanEngine{
    VkInstance vk_instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue graphics_queue;
    uint32_t graphics_queue_family;
    VmaAllocator allocator;
    VkDescriptorPool descriptor_pool;
    APIVersionVulkan api_version{.major=1, .minor=3, .patch=0};

    // These are used in the destructor to delete everything this engine made.

    // The reason why we need to do this instead of just adding destructors is that this way we
    // can ensure that they are destroyed in the correct order, thus the user of the objects does
    // not need to be careful about the order in which they declare the objects. While this usually isn't
    // a concern when they are just declaring them on the stack one by one as the fact that default constructors do not exist for these objects prevents them from initializing
    // an object before they initialize the objects they depend on, they could still call the destructors in the wrong order if they use something like a vector, as those
    // don't need to contain an initialized member.

    // Might also use these in asserts in debug mode to ensure this is the engine that made them.
    // Maybe I'll get around to deleting individual elements
    struct SwapchainTrackingInfo{VkSwapchainKHR swapchain; std::vector<VkImageView> image_views;};
    std::vector<SwapchainTrackingInfo> created_swapchains;
    struct ImageTrackingInfo{VkImage image; VmaAllocation allocation;};
    std::vector<ImageTrackingInfo> created_images;
    std::unordered_set<VkImageView> created_image_views;
    std::vector<VkCommandPool> created_command_pools;
    std::vector<VkFence> created_fences;
    std::vector<VkSemaphore> created_semaphores;
    std::vector<VkDescriptorSetLayout> descriptor_set_layouts_to_delete;
    std::vector<VkShaderModule> created_shader_modules;
    std::vector<VkPipeline> created_pipelines;
    std::vector<VkPipelineLayout> created_pipeline_layouts;
    struct BufferTrackingInfo{VkBuffer buffer; VmaAllocation allocation;};
    std::vector<BufferTrackingInfo> created_buffers;

    bool imgui_is_initialized = false;

    VulkanEngine(const VulkanEngine &) = delete;
    VulkanEngine(VulkanEngine &&) = delete;
    VulkanEngine &operator=(const VulkanEngine &) = delete;
    VulkanEngine &operator=(VulkanEngine &&) = delete;

    VulkanEngine(SDL_Window *window);
    ~VulkanEngine();

    void init_imgui(SDL_Window *window, VkFormat image_format, MSAALevel msaa_level = MSAALevel::OFF);
};

export enum class ImageAspects:VkImageAspectFlags{
    COLOR = VK_IMAGE_ASPECT_COLOR_BIT,
    DEPTH = VK_IMAGE_ASPECT_DEPTH_BIT,
    STENCIL = VK_IMAGE_ASPECT_STENCIL_BIT,
    DEPTH_STENCIL = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
};

export struct Image{
    VkImage vk_image;
    VmaAllocation allocation;
    VkExtent2D extent;
    VkFormat format;
    // Important to remember that this isn't the layout the image is currently in, but is instead
    // the layout that the latest recorded image barrier command (transition command) has transitioned it to
    VkImageLayout layout;

    [[nodiscard]] VkFormat get_format() const { return format; }

    Image(
        VulkanEngine &vk,
        VkExtent2D extent,
        VkFormat format,
        VkImageUsageFlags image_usage_flags,
        VkMemoryPropertyFlagBits memory_property_flags
    );

    Image(VkImage img, VkExtent2D extent, VkFormat format);
};

export struct ImageView{
    VkImageView view;

    ImageView(
        VulkanEngine &vk,
        const Image &img,
        ImageAspects aspects,
        uint32_t base_mip_level,
        uint32_t mip_level_count
    );

    ImageView(VkImageView vk_view);
};

export struct GpuFence{
    VkFence fence;
    GpuFence(VulkanEngine &vk, bool signaled);

    void wait(const VulkanEngine &vk) const;
    [[nodiscard]] bool is_signaled(const VulkanEngine &vk) const;
};

export struct GpuSemaphore{
    VkSemaphore semaphore;
    GpuSemaphore(VulkanEngine &vk);
};

export struct Swapchain{
    VkSwapchainKHR swapchain{};
    VkFormat image_format{};
    VkExtent2D extent{};
    std::vector<Image> images;
    std::vector<ImageView> image_views;

    [[nodiscard]] std::vector<Image> &get_images() { return images; }
    [[nodiscard]] std::vector<ImageView> &get_image_views() { return image_views; }
    [[nodiscard]] const VkExtent2D &get_extent() const { return extent; }
    [[nodiscard]] VkFormat get_format() const { return image_format; }

    Swapchain(VulkanEngine &vk, SDL_Window *window, VkPresentModeKHR present_mode);

    // Returns the index of the next image in the swapchain
    [[nodiscard]] uint32_t acquire_next_image(VulkanEngine &vk, GpuSemaphore signal_sema) const;

    void present(VulkanEngine &vk, GpuSemaphore wait_sema, uint32_t swapchain_image_index);

 private:
    void build_swapchain(
        VulkanEngine &vk,
        SDL_Window *window,
        VkPresentModeKHR present_mode
    );

 public:
    // Almost certainly not to be used outside of the vulkan-util.cppm file. The swapchain is destroyed by the
    // VulkanInstanceInfo that made it.
    void destroy_this_swapchain(VkDevice device);

    void rebuild_swapchain(
        VulkanEngine &vk,
        SDL_Window *window,
        VkPresentModeKHR present_mode
    );
};

export struct CommandPool{
    VkCommandPool pool;
    CommandPool(VulkanEngine &vk);
};

export struct DescriptorSet{
    VkDescriptorSet set;
    VkDescriptorSetLayout layout;

    DescriptorSet(VulkanEngine &vk, VkDescriptorSetLayout layout);

    void update(VulkanEngine &vk, uint32_t bind, std::span<ImageView> views);
    void update(VulkanEngine &vk, uint32_t bind, ImageView &view);
};

struct Shader{
    VkShaderModule module;

    Shader(VulkanEngine &vk, const std::string_view filepath);
};

export struct ComputeShader : public Shader{
    ComputeShader(VulkanEngine &vk, const std::string_view filepath)
    :Shader(vk, filepath) {
        assert(filepath.find(".comp") != std::string_view::npos && "Compute shaders should have .comp in their name. You either passed the wrong shader or your shader does not follow naming convention");
    }
};
export struct VertexShader : public Shader {
    VertexShader(VulkanEngine &vk, const std::string_view filepath)
    :Shader(vk, filepath) {
        assert(filepath.find(".vert") != std::string_view::npos && "Vertex shaders should have .vert in their name. You either passed the wrong shader or your shader does not follow naming convention");
    }
};
export struct FragmentShader : public Shader {
    FragmentShader(VulkanEngine &vk, const std::string_view filepath)
    :Shader(vk, filepath) {
        assert(filepath.find(".frag") != std::string_view::npos && "Fragment shaders should have .frag in their name. You either passed the wrong shader or your shader does not follow naming convention");
    }
};

export struct PipelineLayout{
    VkPipelineLayout layout;

    PipelineLayout(
        VulkanEngine &vk,
        std::initializer_list<DescriptorSet> descriptor_sets,
        const std::optional<std::vector<VkPushConstantRange>> &push_constants
    );
    PipelineLayout(VulkanEngine &vk, DescriptorSet descriptor_set, const std::optional<std::vector<VkPushConstantRange>> &push_constants);
};

// This struct does not need to be kept alive after being used to create a pipeline, and thus can be reset for use in another pipeline's initialization.
export struct SpecializationInfo{
private:
    VkSpecializationInfo info{};
public:
    std::vector<VkSpecializationMapEntry> entries;
    size_t data_buffer_max_capacity_in_bytes{};
    void *data{};
    bool finalized{};
    // Differs from data_buffer_in_bytes in that this is the actually used memory, while data_buffer_in_bytes includes potentially unused memory
    // and always reflects the actual amount of bytes available in the buffer.
    size_t data_size_in_bytes{};

    SpecializationInfo(size_t total_data_size = 128);

    template<typename T>
    SpecializationInfo &add_entry(uint32_t constant_ID, T constant_value){
        static_assert(
            std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t> ||
            std::is_same_v<T, float> ||
            std::is_same_v<T, double> ||
            std::is_same_v<T, VkBool32>,
            "Specialization constants can only be of the following types: int32_t, uint32_t, float, double, VkBool32"
        );
        static_assert(sizeof(T) == 4 || sizeof(T) == 8);

        if(data_size_in_bytes + sizeof(T) > data_buffer_max_capacity_in_bytes){
            assert(false && "Exceeded limit of data buffer while adding entry to specialization info");
            printf("%s", "Exceeded limit of data buffer while adding entry to specialization info");
            abort();
        }
        assert(std::none_of(entries.begin(), entries.end(), [&](const VkSpecializationMapEntry& e){ return e.constantID == constant_ID; }) && "Duplicate specialization constant ID: can't set the same constant twice");

        finalized = false;

        uint32_t offset = data_size_in_bytes;
        entries.emplace_back(constant_ID, offset, sizeof(T));
        memcpy((std::byte*)data + offset, &constant_value, sizeof(T));
        data_size_in_bytes += sizeof(T);
        return *this;
    }

    SpecializationInfo &reset();

    // DO NOT cache the output of this funciton. If you ever reset this instance of SpecializationInfo or let it be destroyed,
    // the returned value of this function becomes invalid. Just pass it to a pipeline creation function and let it go.
    const VkSpecializationInfo *get_vk_specialization_info();

    ~SpecializationInfo();
    SpecializationInfo(const SpecializationInfo&) = delete;
    SpecializationInfo(SpecializationInfo&&) = delete;
    SpecializationInfo& operator=(const SpecializationInfo&) = delete;
    SpecializationInfo& operator=(SpecializationInfo&&) = delete;
};

export struct ComputePipeline{
    VkPipeline pipeline;
    PipelineLayout layout;

    ComputePipeline(
        VulkanEngine &vk,
        ComputeShader shader_module,
        PipelineLayout pipeline_layout,
        SpecializationInfo */* _Nullable */specialization_info = nullptr
    );
};

export struct VulkanBuffer{
    VkBuffer buffer;
    uint32_t capacity_in_bytes;
    VmaAllocation allocation;
private:
    VkDeviceAddress device_address;
public:
    [[nodiscard]] const VkBuffer &get_vk_buffer() const { return buffer; }
    [[nodiscard]] const VkDeviceAddress &get_device_address() const { return device_address; }

    VulkanBuffer(
        VulkanEngine &vk,
        uint32_t capacity_in_bytes,
        VkBufferUsageFlags usage_flags,
        VmaAllocationCreateFlags vma_flags,
        VkMemoryPropertyFlags memory_property_flags_required,
        VkMemoryPropertyFlags memory_property_flags_preferred = 0
    );

    bool operator==(const VulkanBuffer& other) const {
        return buffer == other.buffer;
    }

    bool operator!=(const VulkanBuffer& other) const {
        return !(*this == other);
    }
};

// A GPU-side buffer for the gpu to read from and write to.
export struct StorageBuffer : public VulkanBuffer{
    StorageBuffer(VulkanEngine &vk, uint32_t size_in_bytes, bool is_transfer_source, bool is_transfer_dest);
};

// Used to get data from Host to Device. For performance reasons, host should write into it in sequentially, not at random.
// Copy the contents into a buffer that resides in the Device instead of accessing it from the device directly.
export struct StagingBuffer : public VulkanBuffer{
private:
    void *mapped_data;
public:
    void *get_mapped_data() { return mapped_data; } // remember that if you ever have to add defragmentation or begin unmapping/remapping it you'll have to instead fetch this with vmaGetAllocationInfo every time because it might change

    StagingBuffer(VulkanEngine &vk, uint64_t size_in_bytes);
};

// The opposite of a staging buffer, can be used to copy gpu resources to the cpu or the gpu might store its result in it directly
// It's fine for the host to access it in whatever way it pleases, but access will be slow for the device (unless we're on a UMA machine)
export struct ReadbackBuffer : public VulkanBuffer{
private:
    void *mapped_data;
public:
    void *get_mapped_data() { return mapped_data; } // remember that if you ever have to add defragmentation or begin unmapping/remapping it you'll have to instead fetch this with vmaGetAllocationInfo every time because it might change

    ReadbackBuffer(VulkanEngine &vk, uint32_t size_in_bytes);
};

export struct IndexBuffer : public VulkanBuffer{
    static const VkIndexType index_type = VK_INDEX_TYPE_UINT32;
    IndexBuffer(VulkanEngine &vk, uint32_t total_indexes);
};

export template <typename VertexStruct>
struct VertexBuffer : public VulkanBuffer{
    static const std::size_t vertex_attribute_count = boost::pfr::tuple_size_v<VertexStruct>;
    static const uint32_t stride = sizeof(VertexStruct);

    VertexBuffer(VulkanEngine &vk, uint32_t element_count)
    :VulkanBuffer(vk,
                 sizeof(VertexStruct) * element_count,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 0, 0)
    {
        static_assert(vertex_attribute_count <= 16, "Use only up to 16 vertex attributes as some GPUs don't support more than that.");
    }

    template <typename T>
    static constexpr VkFormat get_vertex_attribute_format(const std::string_view field_name, [[maybe_unused]] const T& field_value){
        using U = std::remove_cvref_t<T>;

        // This list contains only widely supported formats.
        if constexpr      (std::is_same_v<U, int8_norm_t>)   return VK_FORMAT_R8_SNORM;
        else if constexpr (std::is_same_v<U, int16_norm_t>)  return VK_FORMAT_R16_SNORM;
        else if constexpr (std::is_same_v<U, int8_t>)        return VK_FORMAT_R8_SINT;
        else if constexpr (std::is_same_v<U, int16_t>)       return VK_FORMAT_R16_SINT;
        else if constexpr (std::is_same_v<U, int32_t>)       return VK_FORMAT_R32_SINT;
        else if constexpr (std::is_same_v<U, uint8_norm_t>)  return VK_FORMAT_R8_UNORM;
        else if constexpr (std::is_same_v<U, uint16_norm_t>) return VK_FORMAT_R16_UNORM;
        else if constexpr (std::is_same_v<U, uint8_t>)       return VK_FORMAT_R8_UINT;
        else if constexpr (std::is_same_v<U, uint16_t>)      return VK_FORMAT_R16_UINT;
        else if constexpr (std::is_same_v<U, uint32_t>)      return VK_FORMAT_R32_UINT;
        else if constexpr (std::is_same_v<U, float>)         return VK_FORMAT_R32_SFLOAT;
        //else if constexpr (std::is_same_v<U, double>)        return VK_FORMAT_R64_SFLOAT; unsupported by most hardware (~27% support on gpuinfo.org)

        else if constexpr (std::is_same_v<U, ivec2_norm8>)   return VK_FORMAT_R8G8_SNORM;
        else if constexpr (std::is_same_v<U, ivec2_norm16>)  return VK_FORMAT_R16G16_SNORM;
        else if constexpr (std::is_same_v<U, ivec3_norm8>)   return VK_FORMAT_R8G8B8_SNORM; // ~82% support on gpuinfo.org
        else if constexpr (std::is_same_v<U, ivec3_norm16>)  return VK_FORMAT_R16G16B16_SNORM; // ~82.5%
        else if constexpr (std::is_same_v<U, ivec4_norm8>)   return VK_FORMAT_R8G8B8A8_SNORM;
        else if constexpr (std::is_same_v<U, ivec4_norm16>)  return VK_FORMAT_R16G16B16A16_SNORM;
        else if constexpr (std::is_same_v<U, uvec2_norm8>)   return VK_FORMAT_R8G8_UNORM;
        else if constexpr (std::is_same_v<U, uvec2_norm16>)  return VK_FORMAT_R16G16_UNORM;
        else if constexpr (std::is_same_v<U, uvec3_norm8>)   return VK_FORMAT_R8G8B8_UNORM; // ~82%
        else if constexpr (std::is_same_v<U, uvec3_norm16>)  return VK_FORMAT_R16G16B16_UNORM; // ~82%
        else if constexpr (std::is_same_v<U, uvec4_norm8>)   return VK_FORMAT_R8G8B8A8_UNORM;
        else if constexpr (std::is_same_v<U, uvec4_norm16>)  return VK_FORMAT_R16G16B16A16_UNORM;

        else if constexpr (std::is_same_v<U, glm::i8vec2>)   return VK_FORMAT_R8G8_SINT;
        else if constexpr (std::is_same_v<U, glm::i8vec3>)   return VK_FORMAT_R8G8B8_SINT; // ~82%
        else if constexpr (std::is_same_v<U, glm::i8vec4>)   return VK_FORMAT_R8G8B8A8_SINT;
        else if constexpr (std::is_same_v<U, glm::i16vec2>)  return VK_FORMAT_R16G16_SINT;
        else if constexpr (std::is_same_v<U, glm::i16vec3>)  return VK_FORMAT_R16G16B16_SINT; // ~78%
        else if constexpr (std::is_same_v<U, glm::i16vec4>)  return VK_FORMAT_R16G16B16A16_SINT;
        else if constexpr (std::is_same_v<U, glm::ivec2>)    return VK_FORMAT_R32G32_SINT;
        else if constexpr (std::is_same_v<U, glm::ivec3>)    return VK_FORMAT_R32G32B32_SINT;
        else if constexpr (std::is_same_v<U, glm::ivec4>)    return VK_FORMAT_R32G32B32A32_SINT;
        else if constexpr (std::is_same_v<U, glm::u8vec2>)   return VK_FORMAT_R8G8_UINT;
        else if constexpr (std::is_same_v<U, glm::u8vec3>)   return VK_FORMAT_R8G8B8_UINT; // ~82%
        else if constexpr (std::is_same_v<U, glm::u8vec4>)   return VK_FORMAT_R8G8B8A8_UINT;
        else if constexpr (std::is_same_v<U, glm::u16vec2>)  return VK_FORMAT_R16G16_UINT;
        else if constexpr (std::is_same_v<U, glm::u16vec3>)  return VK_FORMAT_R16G16B16_UINT; // ~78%
        else if constexpr (std::is_same_v<U, glm::u16vec4>)  return VK_FORMAT_R16G16B16A16_UINT;
        else if constexpr (std::is_same_v<U, glm::uvec2>)    return VK_FORMAT_R32G32_UINT;
        else if constexpr (std::is_same_v<U, glm::uvec3>)    return VK_FORMAT_R32G32B32_UINT;
        else if constexpr (std::is_same_v<U, glm::uvec4>)    return VK_FORMAT_R32G32B32A32_UINT;

        else if constexpr (std::is_same_v<U, glm::vec2>)     return VK_FORMAT_R32G32_SFLOAT;
        else if constexpr (std::is_same_v<U, glm::vec3>)     return VK_FORMAT_R32G32B32_SFLOAT;
        else if constexpr (std::is_same_v<U, glm::vec4>)     return VK_FORMAT_R32G32B32A32_SFLOAT;

        else if constexpr (std::is_same_v<U, int32_A2R10G10B10_t>)          return VK_FORMAT_A2R10G10B10_SINT_PACK32;
        else if constexpr (std::is_same_v<U, int32_A2R10G10B10_norm_t>)     return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
        else if constexpr (std::is_same_v<U, uint32_A2R10G10B10_t>)         return VK_FORMAT_A2R10G10B10_UINT_PACK32;
        else if constexpr (std::is_same_v<U, uint32_A2R10G10B10_norm_t>)    return VK_FORMAT_A2R10G10B10_UNORM_PACK32;


        std::string name{field_name};
        printf("Your vertex struct conainted a field named %s which has a type that isn't supported as a vertex attribute.", name.c_str());
        assert(false);
        abort();
        return VK_FORMAT_UNDEFINED;
    }

    [[nodiscard]] static constexpr std::array<VkVertexInputAttributeDescription, vertex_attribute_count>
    get_vertex_attribute_descriptions(uint32_t binding = 0, uint32_t first_location = 0)
    {
        std::array<VkVertexInputAttributeDescription, vertex_attribute_count> vertex_attribute_descriptions;
        uint32_t curr_location = first_location;
        uint32_t curr_offset = 0;
        int i = 0;
        boost::pfr::for_each_field_with_name(VertexStruct{}, [&](std::string_view name, const auto &value){
            vertex_attribute_descriptions[i++] = {
                .location = curr_location,
                .binding = binding,
                .format = get_vertex_attribute_format(name, value),
                .offset = curr_offset,
            };
            ++curr_location;
            curr_offset += sizeof(std::remove_cvref_t<decltype(value)>);
        });
        return vertex_attribute_descriptions;
    }
};

export template <typename T>
struct GraphicsPipeline{
    VkPipeline pipeline;
    PipelineLayout layout;
    GraphicsPipeline(
        VulkanEngine &vk,
        VertexShader vert_shader,
        FragmentShader frag_shader,
        PipelineLayout pipeline_layout,
        const VertexBuffer<T> &vertex_buffer,
        VkFormat color_attachment_format,
        VkFormat depth_attachment_format,
        MSAALevel msaa_level,
        SpecializationInfo */* _Nullable */vert_specialization_info = nullptr,
        SpecializationInfo */* _Nullable */frag_specialization_info = nullptr,
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
    :
    layout(pipeline_layout)
    {
        assert(
            topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ||
            topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
            topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        );
        VkPipelineShaderStageCreateInfo shader_stage_infos[2];

        const auto make_pipeline_shader_stage_info = [](
            Shader shader_module,
            VkShaderStageFlagBits stage,
            SpecializationInfo */* _Nullable */specialization_info)
        {
            return VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags{},
                .stage = stage,
                .module = shader_module.module,
                .pName = "main",
                .pSpecializationInfo = specialization_info != nullptr ? specialization_info->get_vk_specialization_info() : nullptr,
            };
        };

        shader_stage_infos[0] = make_pipeline_shader_stage_info(vert_shader, VK_SHADER_STAGE_VERTEX_BIT, vert_specialization_info);
        shader_stage_infos[1] = make_pipeline_shader_stage_info(frag_shader, VK_SHADER_STAGE_FRAGMENT_BIT, frag_specialization_info);

        VkPipelineRenderingCreateInfo pipeline_rendering_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            .viewMask{},
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &color_attachment_format,
            .depthAttachmentFormat = depth_attachment_format,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
        };

        VkVertexInputBindingDescription vertex_binding_description{
            .binding = 0,
            .stride = vertex_buffer.stride,
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        auto vertex_attribute_descriptions = vertex_buffer.get_vertex_attribute_descriptions();

        VkPipelineVertexInputStateCreateInfo vert_input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertex_binding_description,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attribute_descriptions.size()),
            .pVertexAttributeDescriptions = vertex_attribute_descriptions.data(),
        };

        VkPipelineInputAssemblyStateCreateInfo assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .topology = topology,
            .primitiveRestartEnable = VK_FALSE,
        };

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster_state{};
        raster_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster_state.cullMode = VK_CULL_MODE_BACK_BIT;
        raster_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster_state.lineWidth = 1.0f;
        raster_state.polygonMode = VK_POLYGON_MODE_FILL;

        VkPipelineMultisampleStateCreateInfo ms_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .rasterizationSamples = static_cast<VkSampleCountFlagBits>(msaa_level),
            .sampleShadingEnable = VK_FALSE, //true = ssaa, false = msaa
            .minSampleShading{},
            .pSampleMask = nullptr,
            .alphaToCoverageEnable{},
            .alphaToOneEnable{}
        };

        VkPipelineDepthStencilStateCreateInfo ds_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = static_cast<VkBool32>(depth_attachment_format != VK_FORMAT_UNDEFINED),
            .depthWriteEnable = static_cast<VkBool32>(depth_attachment_format != VK_FORMAT_UNDEFINED),
            .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL, // reverse Z. "OR_EQUAL" for when we do z prepass later

            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f
        };

        VkPipelineColorBlendAttachmentState color_blend_attachment_alpha{
            .blendEnable = VK_TRUE,

            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,

            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,

            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo color_blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext =nullptr,
            .flags{},
            .logicOpEnable = VK_FALSE,
            .logicOp{},
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment_alpha,
            .blendConstants{},
        };

        std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags{},
            .dynamicStateCount = dynamic_states.size(),
            .pDynamicStates = dynamic_states.data(),
        };

        VkGraphicsPipelineCreateInfo graphics_pipeline_create_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &pipeline_rendering_create_info,
            .flags{},
            .stageCount = 2,
            .pStages = shader_stage_infos,
            .pVertexInputState = &vert_input_state_info,
            .pInputAssemblyState = &assembly,
            .pTessellationState = nullptr,
            .pViewportState = &viewport_state,
            .pRasterizationState = &raster_state,
            .pMultisampleState = &ms_state,
            .pDepthStencilState = &ds_state,
            .pColorBlendState = &color_blend,
            .pDynamicState = &dynamic_state_info,
            .layout = this->layout.layout,
            .renderPass = VK_NULL_HANDLE,
            .subpass{},
            .basePipelineHandle{},
            .basePipelineIndex{},
        };

        VkResult result = (vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &graphics_pipeline_create_info, nullptr, &this->pipeline));
        if(result != VK_SUCCESS){
            printf("Vulkan error with code: %d", result);
            abort();
        }
        vk.created_pipelines.push_back(pipeline);
    }
};

export struct CommandBuffer{
    VkCommandBuffer buffer;
    CommandBuffer(const VulkanEngine &vk, const CommandPool &pool);
    void draw_imgui(ImageView target_image_view, VkExtent2D draw_extent) const;
    void restart(bool one_time_submit) const;
    void submit(
        VulkanEngine &vk,
        std::optional<GpuSemaphore> wait_sema,
        std::optional<VkPipelineStageFlagBits2> wait_stage_mask, // Commands in the buffer that use these stages will not run until wait_sema is signaled
        std::optional<GpuSemaphore> signal_sema, // Will be signaled when every command in the buffer is complete
        std::optional<VkPipelineStageFlagBits2> signal_stage_mask, // Stages that wait on the signal_sema will have access to writes done by commands in the buffer (other stages will have outdated cached data, causing errors)
        GpuFence signal_fence
    )const;
    void submit(VulkanEngine &vk, GpuFence signal_fence) const;
    void copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, const std::span<VkBufferCopy2> ranges) const;
    void copy_buffer(const VulkanBuffer &src, const VulkanBuffer &dst, VkBufferCopy2 &range) const;
    void copy_entire_buffer(const VulkanBuffer &src, const VulkanBuffer &dst) const;

    template <typename T>
    void draw_indexed(
        ImageView color_attachment,
        ImageView depth_attachment,
        VkExtent2D draw_extent,
        const VertexBuffer<T> &vertex_buffer,
        const IndexBuffer &index_buffer,
        uint32_t index_count) const
    {
        VkRenderingAttachmentInfo color_attachment_info{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext       = nullptr,
            .imageView   = color_attachment.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode{},
            .resolveImageView{},
            .resolveImageLayout{},
            .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{},
        };

        VkRenderingAttachmentInfo depth_attachment_info{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext       = nullptr,
            .imageView   = depth_attachment.view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode{},
            .resolveImageView{},
            .resolveImageLayout{},
            .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{
                .depthStencil{.depth = 0.0f, .stencil = 0}
            },
        };

        const auto make_rendering_info = [](
            VkExtent2D renderExtent,
            VkRenderingAttachmentInfo* colorAttachment,
            VkRenderingAttachmentInfo* depthAttachment)
        {
            return VkRenderingInfo{
                .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext                = nullptr,
                .flags{},
                .renderArea           = {{0, 0}, renderExtent},
                .layerCount           = 1,
                .viewMask{},
                .colorAttachmentCount = 1,
                .pColorAttachments    = colorAttachment,
                .pDepthAttachment     = depthAttachment,
                .pStencilAttachment   = nullptr,
            };
        };

        VkRenderingInfo rendering_info = make_rendering_info(draw_extent, &color_attachment_info, &depth_attachment_info);

        vkCmdBeginRendering(buffer, &rendering_info);

        VkDeviceSize offsets = 0;
        vkCmdBindVertexBuffers(buffer, 0, 1, &vertex_buffer.buffer, &offsets);
        vkCmdBindIndexBuffer(buffer, index_buffer.buffer, 0, IndexBuffer::index_type);

        VkViewport viewport = {
            .x = 0,
            .y = 0,
            .width = static_cast<float>(draw_extent.width),
            .height = static_cast<float>(draw_extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(buffer, 0, 1, &viewport);

        VkRect2D scissor{
            .offset{.x = 0, .y = 0},
            .extent{draw_extent},
        };
        vkCmdSetScissor(buffer, 0, 1, &scissor);

        vkCmdDrawIndexed(buffer, index_count, 1, 0, 0, 0);
        vkCmdEndRendering(buffer);
    }

    struct BarrierInfo{
        Image &img;
        bool discard_current_data;
        VkImageLayout new_layout;
        VkPipelineStageFlags2 src_stage_mask;
        VkAccessFlags2 src_access_mask;
        VkPipelineStageFlags2 dst_stage_mask;
        VkAccessFlags2 dst_access_mask;
        ImageAspects aspects;
        uint32_t base_mip_level = 0;
        uint32_t mip_level_count = VK_REMAINING_MIP_LEVELS;
    };
    template<typename... BarrierInfo_T>
    void barrier(const BarrierInfo_T&... barrier_infos) const {
        static_assert((std::is_same_v<std::remove_cvref_t<BarrierInfo_T>, BarrierInfo> && ...), "Arguments to CommandBuffer::barrier must be CommandBuffer::BarrierInfo types.");
        std::array<VkImageMemoryBarrier2, sizeof...(BarrierInfo_T)> image_barriers;

        int i = 0;
        const auto add_image_barrier = [&](const BarrierInfo& barrier_info){
            image_barriers[i++] = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = barrier_info.src_stage_mask,
                .srcAccessMask = barrier_info.src_access_mask,
                .dstStageMask = barrier_info.dst_stage_mask,
                .dstAccessMask = barrier_info.dst_access_mask,
                .oldLayout = barrier_info.discard_current_data ? VK_IMAGE_LAYOUT_UNDEFINED : barrier_info.img.layout,
                .newLayout = barrier_info.new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // todo was 0 before and worked, figure out more about queue families
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = barrier_info.img.vk_image,
                .subresourceRange = {
                    .aspectMask     = static_cast<VkImageAspectFlags>(barrier_info.aspects),
                    .baseMipLevel   = barrier_info.base_mip_level,
                    .levelCount     = barrier_info.mip_level_count,
                    .baseArrayLayer = 0,
                    .layerCount     = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            barrier_info.img.layout = barrier_info.new_layout;
        };

        (add_image_barrier(barrier_infos), ...);

        VkDependencyInfo dep_info {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags{},
            .memoryBarrierCount{},
            .pMemoryBarriers{},
            .bufferMemoryBarrierCount{},
            .pBufferMemoryBarriers{},
            .imageMemoryBarrierCount = static_cast<uint32_t>(sizeof...(BarrierInfo_T)),
            .pImageMemoryBarriers = image_barriers.data(),
        };

        vkCmdPipelineBarrier2(this->buffer, &dep_info);
    }

    void blit(
        const Image &source,
        const Image &destination,
        glm::ivec2 src_top_left,
        glm::ivec2 src_bottom_right,
        glm::ivec2 dst_top_left,
        glm::ivec2 dst_bottom_right,
        ImageAspects aspects
    )const; // todo add VkCmdCopyImage function for when we don't need to blit

    void blit_entire_images(const Image &source, const Image &destination, ImageAspects aspects) const;
    void bind_pipeline(const ComputePipeline &pipeline) const;

    template <typename T>
    void bind_pipeline(const GraphicsPipeline<T> &pipeline) const{
        vkCmdBindPipeline(this->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    }

    void bind_descriptor_sets(const ComputePipeline &pipeline, std::initializer_list<DescriptorSet> sets) const;
    void bind_descriptor_sets(ComputePipeline pipeline, DescriptorSet set) const;

    template <typename T>
    void update_push_constants(
        ComputePipeline pipeline,
        const PushConstant<T>& push_constant)
    {
        assert(push_constant.shader_stages & VK_SHADER_STAGE_COMPUTE_BIT
               && "Updated push constants for a compute pipeline with a push constant that cannot be accessed by compute shaders.");
        vkCmdPushConstants(this->buffer,
                           pipeline.layout.layout,
                           push_constant.shader_stages,
                           push_constant.offset,
                           push_constant.size,
                           &push_constant.data);
    }

    template <typename T, typename U>
    void update_push_constants(
        GraphicsPipeline<T> pipeline,
        const PushConstant<U>& push_constant)
    {
        assert(push_constant.shader_stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
               && "Updated push constants for a graphics pipeline with a push constant that cannot be accessed by vertex nor fragment shaders.");
        vkCmdPushConstants(this->buffer,
                           pipeline.layout.layout,
                           push_constant.shader_stages,
                           push_constant.offset,
                           push_constant.size,
                           &push_constant.data);
    }

    void dispatch(uint32_t x, uint32_t y, uint32_t z) const;
    void end() const;
};

// This is what you're meant to use instead of Descriptor Set Layouts and Descriptor Set Layout Bindings. You can
// keep using build as many times as you want just like you would with a Layout.
export struct DescriptorSetBuilder{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayout layout{};
    bool finalized = false;

    DescriptorSetBuilder &bind(uint32_t binding, VkDescriptorType type, VkShaderStageFlagBits accessible_stages_flags = VK_SHADER_STAGE_ALL);

    DescriptorSet build(VulkanEngine &vk);

    // Don't reset unless you know you won't be making another descriptor set with the same layout again. It's slower otherwise.
    // Objects passed into the creation of other objects must not be destroyed while the new object is in use, so we cannot destroy the layout here.
    // Since we use all descriptor sets until the end of the program, there is no reason to want that, either.
    DescriptorSetBuilder &reset();
};

