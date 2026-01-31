#ifndef VKML_COMPILER_COMP_HPP
#define VKML_COMPILER_COMP_HPP


#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Location.h>

#include <typeinfo>
#include <cstddef>

// Implementation details for Tensor utilities
namespace tensor_detail {

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


namespace compiler {

	struct Context {
		mlir::MLIRContext ctx;
		mlir::ModuleOp mod;	
	};

	static void init(Context& context) {
		// Load necessary dialects
		mlir::DialectRegistry registry;
		mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
		mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
		mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
		mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
		mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
		mlir::func::registerInlinerExtension(registry);
		// mlir::tosaOps::registerBufferizableOpInterfaceExternalModels(registry);
		context.ctx.appendDialectRegistry(registry);

		// Load all the dialects
		context.ctx.loadDialect<mlir::func::FuncDialect>();
		context.ctx.loadDialect<mlir::tosa::TosaDialect>();
		context.ctx.loadDialect<mlir::gpu::GPUDialect>();
		context.ctx.loadDialect<mlir::arith::ArithDialect>();
		context.ctx.loadDialect<mlir::scf::SCFDialect>();
		context.ctx.loadDialect<mlir::tensor::TensorDialect>();
		context.ctx.loadDialect<mlir::memref::MemRefDialect>();
		context.ctx.loadDialect<mlir::bufferization::BufferizationDialect>();
		context.ctx.loadDialect<mlir::linalg::LinalgDialect>();
		context.ctx.loadDialect<mlir::math::MathDialect>();
		context.ctx.loadDialect<mlir::spirv::SPIRVDialect>();

		context.mod =  mlir::ModuleOp::create(mlir::UnknownLoc::get(&context.ctx));
	}

	static Context *instance = NULL;

	static Context* getInstance() {
		if (instance == NULL) {
			instance = new Context();
			init(*instance);
		}
		return instance;
	}

	static void freeInstance() {
		if (instance != NULL) {
			delete instance;
			instance = NULL;
		}
	}

	mlir::UnrankedTensorType buildTensorType(Context& context, const std::type_info& typeInfo) {
		return mlir::UnrankedTensorType::get(tensor_detail::cToMLIRType(&context.ctx, typeInfo));
	}
	
	mlir::RankedTensorType buildTensorType(Context& context, const llvm::ArrayRef<long int>& shape, const std::type_info& typeInfo) {
		return mlir::RankedTensorType::get(shape, tensor_detail::cToMLIRType(&context.ctx, typeInfo));
	}
	
	mlir::func::FuncOp createFunctionWithTosaOp(Context& context, const llvm::StringRef name, const mlir::Type& input1, const mlir::Type& input2, const mlir::Type& output) {

		llvm::SmallVector<mlir::Type> inputTypes{input1, input2};
		llvm::SmallVector<mlir::Type> resultTypes { output };

		mlir::OpBuilder builder(&context.ctx);
		auto funcType = builder.getFunctionType(inputTypes, resultTypes);
		auto funcOp = mlir::func::FuncOp::create(builder.getUnknownLoc(), name, funcType);

		builder.setInsertionPointToStart(funcOp.addEntryBlock());
		auto tosaOp = mlir::tosa::AddOp::create(builder, funcOp.getLoc(), resultTypes, funcOp.getArgument(0), funcOp.getArgument(1));
		builder.create<mlir::func::ReturnOp>(funcOp.getLoc(), tosaOp.getResult());
		
		context.mod.push_back(funcOp);

		return funcOp;
	}

	template<typename T1, typename T2>
	mlir::func::FuncOp createFunctionWithTosaOp(Context &context, const llvm::StringRef name, std::type_info const& type1 = typeid(T1), std::type_info const& type2 = typeid(T2)) {
		auto input1 = buildTensorType(context, type1);
		auto input2 = buildTensorType(context, type2);
		auto output = buildTensorType(context, typeid(std::common_type_t<T1, T2>));
		return createFunctionWithTosaOp(context, name, input1, input2, output);
	}

	template<typename T1, typename T2>
	mlir::func::FuncOp createFunctionWithTosaOp(Context &context, const llvm::StringRef name, const llvm::ArrayRef<long int>& shape1, const llvm::ArrayRef<long int>& shape2, std::type_info const& type1 = typeid(T1), std::type_info const& type2 = typeid(T2)) {
		auto input1 = buildTensorType(context, shape1, type1);
		auto input2 = buildTensorType(context, shape2, type2);
		// needs breoadcast logic here
		
		auto output = buildTensorType(context, shape1, typeid(std::common_type_t<T1, T2>));
		return createFunctionWithTosaOp(context, name, input1, input2, output

	

} // namespace compiler


#endif // VKML_COMPILER_COMP_HPP