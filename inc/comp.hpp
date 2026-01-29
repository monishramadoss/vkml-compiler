#pragma once


// Implementation details for Tensor utilities
namespace tensor_detail {

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

	template <typename U>
	struct is_character_or_byte_or_bool
		: std::bool_constant<
		std::is_same_v<U, char> || std::is_same_v<U, unsigned char> ||
		std::is_same_v<U, signed char> || std::is_same_v<U, std::byte> ||
		std::is_same_v<U, bool>> {};

	static auto cToMLIRType = [](mlir::MLIRContext* ctx, const std::type_info& type) -> mlir::Type {
		if (type == typeid(float)) {
			return mlir::Float32Type::get(ctx);
		}
		else if (type == typeid(double)) {
			return mlir::Float64Type::get(ctx);
		}
		else if (type == typeid(char)) {
			return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
		}
		else if (type == typeid(unsigned char)) {
			return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
		}
		else if (type == typeid(int32_t)) {
			return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
		}
		else if (type == typeid(int64_t)) {
			return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
		}
		else if (type == typeid(uint32_t)) {
			return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
		}
		else if (type == typeid(uint64_t)) {
			return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
		}
		else if (type == typeid(bool)) {
			return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Unsigned);
		}
		else {
			throw std::invalid_argument("Unsupported type for MLIR conversion");
		}
	};

} // namespace tensor_detail


namespace tensor_operations {


#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>


#include <mlir/IR/Module.h>
#include <mlir/IR/MLIRContext.h>

	struct CompilerContext {
		mlir::MLIRContext ctx;
		CompilerContext() {
			// Load necessary dialects
			    mlir::DialectRegistry registry;
				mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
				mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
				mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
				mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
				mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
				mlir::func::registerInlinerExtension(registry);
				ctx.appendDialectRegistry(registry);

				// Load all the dialects
				ctx.loadDialect<mlir::func::FuncDialect>();
				ctx.loadDialect<mlir::ml_program::MLProgramDialect>();
				ctx.loadDialect<mlir::gpu::GPUDialect>();
				ctx.loadDialect<mlir::arith::ArithDialect>();
				ctx.loadDialect<mlir::scf::SCFDialect>();
				ctx.loadDialect<mlir::tensor::TensorDialect>();
				ctx.loadDialect<mlir::memref::MemRefDialect>();
				ctx.loadDialect<mlir::bufferization::BufferizationDialect>();
				ctx.loadDialect<mlir::linalg::LinalgDialect>();
				ctx.loadDialect<mlir::math::MathDialect>();
				ctx.loadDialect<mlir::spirv::SPIRVDialect>();
				ctx.loadDialect<mlir::cf::ControlFlowDialect>();
				ctx.loadDialect<mlir::DLTIDialect>();
		}


	}


} // namespace tensor_operations