// Simple test to verify IREE integration
#include "iree/runtime/api.h"
#include <iostream>

int main() {
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    
    iree_runtime_instance_t* instance = NULL;
    iree_status_t status = iree_runtime_instance_create(
        &instance_options,
        iree_allocator_system(),
        &instance);
    
    if (iree_status_is_ok(status)) {
        std::cout << "IREE runtime instance created successfully!" << std::endl;
        iree_runtime_instance_release(instance);
        return 0;
    } else {
        std::cerr << "Failed to create IREE runtime instance" << std::endl;
        iree_status_ignore(status);
        return 1;
    }
}
