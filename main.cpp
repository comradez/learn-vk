#define VK_USE_PLATFORM_XLIB_KHR
#include <X11/Xlib.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <limits>
#include <vector>

constexpr uint32_t WIDTH = 800u;
constexpr uint32_t HEIGHT = 600u;
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2u;

static constexpr std::array validation_layers{"VK_LAYER_KHRONOS_validation"};
static constexpr std::array device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
static constexpr std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

#ifdef NDEBUG
constexpr bool enable_validation_layer = false;
#else
constexpr bool enable_validation_layer = true;
#endif

auto checkValidationLayerSupport() -> bool {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    auto availableLayers = std::vector<VkLayerProperties>(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, availableLayers.data());

    for (const auto layer_name : validation_layers) {
        if (!std::any_of(availableLayers.cbegin(), availableLayers.cend(),
                         [&](const VkLayerProperties &available_layer) {
                             return std::strcmp(available_layer.layerName,
                                                layer_name) == 0;
                         })) {
            return false;
        }
    }
    return true;
}

#ifndef NDEBUG
auto CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) -> VkResult {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

auto DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debug_messenger,
                                   const VkAllocationCallbacks *pAllocator)
    -> void {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debug_messenger, pAllocator);
    }
}
#endif

auto read_file(const char *filename) -> std::vector<char> {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    auto file_size = file.tellg();
    auto buffer = std::vector<char>(file_size);
    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();
    return buffer;
}

class HelloTriangleApplication {
public:
    void run() {
        init_window();
        init_vulkan();
        main_loop();
        cleanup();
    }

private:
    GLFWwindow *window;

    VkInstance instance;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphics_queue;
    VkQueue present_queue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swap_chain;
    std::vector<VkImage> swap_chain_images;
    std::vector<VkImageView> swap_chain_image_views;
    VkFormat swap_chain_image_format;
    VkExtent2D swap_chain_extent;
    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;
    std::vector<VkFramebuffer> swap_chain_framebuffers;
    VkCommandPool command_pool;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> command_buffers;
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> image_available_semaphores;
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> render_finished_semaphores;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> in_flight_fences;
    uint32_t current_frame = 0u;
    bool framebuffer_resized = false;

#ifndef NDEBUG
    VkDebugUtilsMessengerEXT debug_messenger;
#endif

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphics_family;
        std::optional<uint32_t> present_family;

        auto is_complete() const -> bool {
            return graphics_family.has_value() && present_family.has_value();
        }
    };

    auto init_window() -> void {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
    }

    static auto framebuffer_resize_callback(GLFWwindow *window, int width, int height) -> void {
        auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
        app->framebuffer_resized = true;
    }

    auto init_vulkan() -> void {
        create_instance();
        if constexpr (enable_validation_layer) {
            setup_debug_messenger();
        }
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swap_chain();
        create_image_views();
        create_render_pass();
        create_graphics_pipeline();
        create_framebuffers();
        create_command_pool();
        create_command_buffer();
        create_sync_objects();
    }

    auto create_sync_objects() -> void {
        auto semaphore_info = VkSemaphoreCreateInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        auto fence_info = VkFenceCreateInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores!");
            }
        }
    }

    auto record_command_buffer(VkCommandBuffer buffer, uint32_t image_index) -> void {
        auto begin_info = VkCommandBufferBeginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        if (vkBeginCommandBuffer(buffer, &begin_info) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        auto clear_color = VkClearValue{{{0.0f, 0.0f, 0.0f, 1.0f}}};
        auto render_pass_begin_info = VkRenderPassBeginInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = swap_chain_framebuffers[image_index],
            .renderArea {
                .offset {0, 0},
                .extent = swap_chain_extent,
            },
            .clearValueCount = 1u,
            .pClearValues = &clear_color,
        };
        vkCmdBeginRenderPass(buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline); 
            auto viewport = VkViewport {
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(swap_chain_extent.width),
                .height = static_cast<float>(swap_chain_extent.height),
                .minDepth = 0.0f, 
                .maxDepth = 1.0f,
            };
            vkCmdSetViewport(buffer, 0, 1, &viewport);

            auto scissor = VkRect2D {
                .offset = {0, 0},
                .extent = swap_chain_extent,
            };
            vkCmdSetScissor(buffer, 0, 1, &scissor);
            vkCmdDraw(buffer, 3, 1, 0, 0); // 我的三角形！
        vkCmdEndRenderPass(buffer);
        if (vkEndCommandBuffer(buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    auto create_command_buffer() -> void {
        auto alloc_info = VkCommandBufferAllocateInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(command_buffers.size()),
        };
        if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    auto create_command_pool() -> void {
        auto queue_family_indices = find_queue_families(physical_device);
        auto pool_info = VkCommandPoolCreateInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_indices.graphics_family.value(),
        };
        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    auto create_framebuffers() -> void {
        swap_chain_framebuffers.resize(swap_chain_image_views.size());
        for (auto i = 0u; i < swap_chain_framebuffers.size(); i++) {
            auto framebuffer_info = VkFramebufferCreateInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = render_pass,
                .attachmentCount = 1u,
                .pAttachments = &swap_chain_image_views[i],
                .width = swap_chain_extent.width,
                .height = swap_chain_extent.height,
                .layers = 1u,
            };
            if (vkCreateFramebuffer(device, &framebuffer_info, nullptr, &swap_chain_framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    auto create_render_pass() -> void {
        auto color_attachment = VkAttachmentDescription{
            .format = swap_chain_image_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        auto color_attachment_ref = VkAttachmentReference{
            .attachment = 0u,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        auto subpass = VkSubpassDescription{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1u,
            .pColorAttachments = &color_attachment_ref,
        };
        auto dependency = VkSubpassDependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0u,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0u,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        auto render_pass_info = VkRenderPassCreateInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1u,
            .pAttachments = &color_attachment,
            .subpassCount = 1u,
            .pSubpasses = &subpass,
            .dependencyCount = 1u,
            .pDependencies = &dependency,
        };
        if (vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    auto create_graphics_pipeline() -> void {
        auto vertex_shader = read_file("shaders/vert.spv");
        auto fragment_shader = read_file("shaders/frag.spv");

        auto create_shader_module = [&](const std::vector<char> &code) -> VkShaderModule {
            auto create_info = VkShaderModuleCreateInfo{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = code.size(),
                .pCode = reinterpret_cast<const uint32_t *>(code.data()),
            };
            auto shader_module = VkShaderModule{};
            if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create shader module.");
            }
            return shader_module;
        };

        auto vert_shader_module = create_shader_module(vertex_shader);
        auto frag_shader_module = create_shader_module(fragment_shader);

        auto vert_shader_stage_info = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_shader_module,
            .pName = "main",
        };
        auto frag_shader_stage_info = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_shader_module,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

        auto vertex_input_info = VkPipelineVertexInputStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0u,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0u,
            .pVertexAttributeDescriptions = nullptr,
        };

        auto input_assembly = VkPipelineInputAssemblyStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        auto viewport = VkViewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = (float)swap_chain_extent.width,
            .height = (float)swap_chain_extent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        auto scissor = VkRect2D{
            .offset = {0, 0},
            .extent = swap_chain_extent,
        };
        auto dynamic_state = VkPipelineDynamicStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
        };
        auto viewport_state = VkPipelineViewportStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1u,
            .scissorCount = 1u,
        };

        auto rasterizer = VkPipelineRasterizationStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.f,
        };

        auto multisampling = VkPipelineMultisampleStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };

        auto color_blend_attachment = VkPipelineColorBlendAttachmentState{
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        auto color_blending = VkPipelineColorBlendStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1u,
            .pAttachments = &color_blend_attachment,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
        };

        auto pipeline_layout_info = VkPipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        };
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        auto pipeline_info = VkGraphicsPipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2u,
            .pStages = shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = nullptr,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout,
            .renderPass = render_pass,
            .subpass = 0u,
        };
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1u, &pipeline_info, nullptr, &graphics_pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, frag_shader_module, nullptr);
        vkDestroyShaderModule(device, vert_shader_module, nullptr);
    }

    auto create_image_views() -> void {
        swap_chain_image_views.resize(swap_chain_images.size());
        for (auto i = 0u; i < swap_chain_image_views.size(); i++) {
            auto create_info = VkImageViewCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swap_chain_images[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swap_chain_image_format,
                .components{
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            if (vkCreateImageView(device, &create_info, nullptr, &swap_chain_image_views[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views.");
            }
        }
    }

    auto cleanup_swap_chain() -> void {
        for (auto &framebuffer : swap_chain_framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        for (auto &image_view : swap_chain_image_views) {
            vkDestroyImageView(device, image_view, nullptr);
        }
        vkDestroySwapchainKHR(device, swap_chain, nullptr);
    }

    auto recreate_swap_chain() -> void {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);

        cleanup_swap_chain();

        create_swap_chain();
        create_image_views();
        create_framebuffers();
    }

    auto create_swap_chain() -> void {
        auto swap_chain_support = query_swap_chain_support(physical_device);
        auto choose_swap_surface_format = [](const std::vector<VkSurfaceFormatKHR> &availableFormats) {
            for (const auto &availableFormat : availableFormats) {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return availableFormat;
                }
            }
            return availableFormats[0];
        };
        auto choose_swap_present_mode = [](const std::vector<VkPresentModeKHR> &availablePresentModes) {
            for (const auto &availablePresentMode : availablePresentModes) {
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentMode;
                }
            }
            return VK_PRESENT_MODE_FIFO_KHR;
        };
        auto choose_swap_extent = [&](const VkSurfaceCapabilitiesKHR &capabilities) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            } else {
                int width, height;
                glfwGetFramebufferSize(window, &width, &height);
                auto actual_extent = VkExtent2D{
                    .width = static_cast<uint32_t>(width),
                    .height = static_cast<uint32_t>(height),
                };
                actual_extent.width = std::clamp(actual_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
                actual_extent.height = std::clamp(actual_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
                return actual_extent;
            }
        };
        auto surface_format = choose_swap_surface_format(swap_chain_support.formats);
        auto present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
        auto extent = choose_swap_extent(swap_chain_support.capabilities);

        auto image_count = swap_chain_support.capabilities.minImageCount + 1u;
        if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount) {
            image_count = swap_chain_support.capabilities.maxImageCount;
        }

        auto create_info = VkSwapchainCreateInfoKHR{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = image_count,
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1u,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = swap_chain_support.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };

        auto indices = find_queue_families(physical_device);
        auto queue_family_indices = std::array{
            indices.graphics_family.value(),
            indices.present_family.value(),
        };
        if (indices.graphics_family != indices.present_family) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queue_family_indices.data();
        } else {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            create_info.queueFamilyIndexCount = 0;
            create_info.pQueueFamilyIndices = nullptr;
        }

        if (vkCreateSwapchainKHR(device, &create_info, nullptr, &swap_chain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swap_chain, &image_count, nullptr);
        swap_chain_images.resize(image_count);
        vkGetSwapchainImagesKHR(device, swap_chain, &image_count, swap_chain_images.data());
        swap_chain_image_format = surface_format.format;
        swap_chain_extent = extent;
    }

    auto create_surface() -> void {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    auto find_queue_families(const VkPhysicalDevice &device) -> QueueFamilyIndices {
        auto indices = QueueFamilyIndices{};

        uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                                 nullptr);
        auto queue_families = std::vector<VkQueueFamilyProperties>(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                                 queue_families.data());

        for (auto idx = 0u; idx < queue_family_count; idx++) {
            if (queue_families[idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphics_family = idx;
            }
            auto present_support = VkBool32{false};
            vkGetPhysicalDeviceSurfaceSupportKHR(device, idx, surface, &present_support);
            if (present_support) {
                indices.present_family = idx;
            }
            if (indices.is_complete()) {
                break;
            }
        }

        return indices;
    };

    auto create_logical_device() -> void {

        auto indices = find_queue_families(physical_device);
        auto unique_queue_familes = std::set<uint32_t>{
            indices.graphics_family.value(),
            indices.present_family.value(),
        };
        auto queue_priority = 1.0f;

        auto queue_create_infos = std::vector<VkDeviceQueueCreateInfo>{};
        for (auto queue_family : unique_queue_familes) {
            auto queue_creation_info = VkDeviceQueueCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queue_family,
                .queueCount = 1,
                .pQueuePriorities = &queue_priority,
            };
            queue_create_infos.push_back(queue_creation_info);
        }

        auto device_features = VkPhysicalDeviceFeatures{};
        auto creation_info = VkDeviceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size()),
            .pQueueCreateInfos = queue_create_infos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(device_extensions.size()),
            .ppEnabledExtensionNames = device_extensions.data(),
            .pEnabledFeatures = &device_features,
        };
        if constexpr (enable_validation_layer) {
            creation_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
            creation_info.ppEnabledLayerNames = validation_layers.data();
        } else {
            creation_info.enabledLayerCount = 0u;
        }

        if (vkCreateDevice(physical_device, &creation_info, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphics_family.value(), 0, &graphics_queue);
        vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
    }

    auto pick_physical_device() -> void {
        uint32_t device_count;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        if (device_count == 0) {
            throw std::runtime_error("No graphics device with vulkan support found.");
        }
        auto devices = std::vector<VkPhysicalDevice>(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

        auto is_device_suitable = [&](VkPhysicalDevice device) -> bool {
            auto check_device_extension_support = [](VkPhysicalDevice device) -> bool {
                uint32_t extension_count;
                vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

                auto available_extensions = std::vector<VkExtensionProperties>(extension_count);
                vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

                for (const auto &required_extension : device_extensions) {
                    if (!std::any_of(available_extensions.cbegin(), available_extensions.cend(), [&](const VkExtensionProperties &available_extension) {
                            return std::strcmp(available_extension.extensionName, required_extension) == 0;
                        })) {
                        return false;
                    }
                }
                return true;
            };

            auto indices = find_queue_families(device);
            auto extensions_supported = check_device_extension_support(device);
            auto swap_chain_adequate = false;
            if (extensions_supported) {
                auto swap_chain_support = query_swap_chain_support(device);
                swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
            }
            return indices.is_complete() && extensions_supported && swap_chain_adequate;
        };

        physical_device = [&]() -> VkPhysicalDevice {
            for (const auto device : devices) {
                if (is_device_suitable(device)) {
                    return device;
                }
            }
            return VK_NULL_HANDLE;
        }();

        if (physical_device == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> present_modes;
    };

    auto query_swap_chain_support(VkPhysicalDevice device) -> SwapChainSupportDetails {
        auto details = SwapChainSupportDetails{};

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
        if (format_count != 0) {
            details.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, &details.formats[0]);
        }

        uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);
        if (present_mode_count != 0) {
            details.present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
        }

        return details;
    };

    auto populate_debug_messenger_create_info(
        VkDebugUtilsMessengerCreateInfoEXT &create_info) {
        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create_info.pfnUserCallback = debug_callback;
    }

    auto setup_debug_messenger() -> void {
        auto create_info = VkDebugUtilsMessengerCreateInfoEXT{};
        populate_debug_messenger_create_info(create_info);
        if (CreateDebugUtilsMessengerEXT(instance, &create_info, nullptr,
                                         &debug_messenger) != VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug messenger!");
        }
    }

    auto main_loop() -> void {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            draw_frame();
        }
        vkDeviceWaitIdle(device);
    }

    auto draw_frame() -> void {
        vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, std::numeric_limits<uint64_t>::max());
        
        uint32_t image_index;
        auto result = vkAcquireNextImageKHR(device, swap_chain, std::numeric_limits<uint64_t>::max(), image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
            framebuffer_resized = true;
            recreate_swap_chain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to acquire swap chain image.");
        }

        vkResetFences(device, 1, &in_flight_fences[current_frame]);
        vkResetCommandBuffer(command_buffers[current_frame], 0);
        record_command_buffer(command_buffers[current_frame], image_index);

        VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        auto submit_info = VkSubmitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1u,
            .pWaitSemaphores = image_available_semaphores.data() + current_frame,
            .pWaitDstStageMask = wait_stages,
            .commandBufferCount = 1u,
            .pCommandBuffers = command_buffers.data() + current_frame,
            .signalSemaphoreCount = 1u,
            .pSignalSemaphores = render_finished_semaphores.data() + current_frame,
        };
        if (vkQueueSubmit(graphics_queue, 1u, &submit_info, in_flight_fences[current_frame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        auto present_info = VkPresentInfoKHR {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1u,
            .pWaitSemaphores = render_finished_semaphores.data() + current_frame,
            .swapchainCount = 1u,
            .pSwapchains = &swap_chain,
            .pImageIndices = &image_index,
            .pResults = nullptr,
        };
        vkQueuePresentKHR(present_queue, &present_info);
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    auto cleanup() -> void {
        cleanup_swap_chain();
        for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyFence(device, in_flight_fences[i], nullptr);
            vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
            vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        }
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyPipeline(device, graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyRenderPass(device, render_pass, nullptr);
        vkDestroyDevice(device, nullptr);
        if constexpr (enable_validation_layer) {
            DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
        }
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    auto create_instance() -> void {
        if constexpr (enable_validation_layer) {
            if (!checkValidationLayerSupport()) {
                throw std::runtime_error(
                    "validation layers requested, but not available!");
            }
        }

        auto app_info = VkApplicationInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(0, 0, 1),
            .apiVersion = VK_API_VERSION_1_0,
        };

        auto extensions = [&]() -> std::vector<const char *> {
            uint32_t glfw_extension_count;
            auto glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
            auto extensions = std::vector<const char *>(
                glfw_extensions, glfw_extensions + glfw_extension_count);

            if constexpr (enable_validation_layer) {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }

            return extensions;
        }();

        auto create_info = VkInstanceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        auto debug_create_info = VkDebugUtilsMessengerCreateInfoEXT{};
        // If this loc is put outside of the if scope, it works perfectly fine.
        if constexpr (enable_validation_layer) {
            create_info.enabledLayerCount =
                static_cast<uint32_t>(validation_layers.size());
            create_info.ppEnabledLayerNames = validation_layers.data();
            // auto debug_create_info = VkDebugUtilsMessengerCreateInfoEXT {};
            // But if it is put here right before the
            // populate_debug_messenger_create_info, it crashes. I have no idea why.
            populate_debug_messenger_create_info(debug_create_info);
            create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debug_create_info;
        } else {
            create_info.enabledLayerCount = 0;
            create_info.pNext = nullptr;
        }

        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                  void *pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

auto main() -> int {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
