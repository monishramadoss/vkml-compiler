// #include "compiler.h"

// int main() {
//     vkml::Compiler compiler;
//     vkml::TensorOps tensorOps(compiler);
//     vkml::TosaOps tosaOps(compiler);

//     // Create operations in logical order - they'll be stacked automatically
//     auto t1 = tensorOps.createEmptyOp(
//         mlir::Float32Type::get(compiler.getContext()),
//         {2, 3}
//     );

//     auto t2 = tensorOps.createEmptyOp(
//         mlir::Float32Type::get(compiler.getContext()),
//         {2, 3}
//     );

//     // Create TOSA add operation
//     auto add = tosaOps.createAddOp(
//         t1->getResult(0).getType(),
//         t1->getResult(0),
//         t2->getResult(0)
//     );
    
//     // Run the transformation passes
//     compiler.runPasses();

//     return 0;
// }

#include "inc/comp.hpp"

int main() {
    auto* context = compiler::getInstance();

    auto input1 = compiler::buildTensorType(*context, typeid(float));
    auto input2 = compiler::buildTensorType(*context, typeid(float));
    auto funcOp = compiler::createFunctionWithTosaOp<float, float>(*context, "my_function");

    context->mod.dump();

    compiler::freeInstance();
    return 0;
 }