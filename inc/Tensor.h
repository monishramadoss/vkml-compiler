#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "Compiler.h"

template <typename T> class Tensor {
private:
  std::vector<int64_t> shapeStorage_; // Owns the shape memory
  mlir::ArrayRef<int64_t> shape_;     // View into owned storage
  mlir::Operation *src_ = nullptr;
  bool external_ = false;

  std::string symbolic_id_;
  mlir::RankedTensorType type_;
  std::shared_ptr<T> data_;

  mutable mlir::Value tensorValue_;
  mutable bool isInitialized_ = false;

public:
  // Convenience constructor to disambiguate brace-init usage
  Tensor(std::initializer_list<int64_t> dims)
      : Tensor(mlir::ArrayRef<int64_t>(dims.begin(), dims.size())) {}

  Tensor(const mlir::ArrayRef<int64_t> &shape)
      : shapeStorage_(shape.begin(), shape.end()), shape_(shapeStorage_),
        src_(nullptr),
        type_(mlir::RankedTensorType::get(
            shape_,
            tensor_detail::cToMLIRType(
                vkml::Compiler::getInstance()->getContext(), typeid(T)))),
        data_(nullptr), tensorValue_(nullptr), isInitialized_(false) {

    static int id_counter = 0;
    symbolic_id_ = "tensor_" + std::to_string(id_counter++);
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    
    // Create tensor.empty operation
    auto scope = compiler->inMainBeforeTerminator();
    tensorValue_ = builder.create<mlir::tensor::EmptyOp>(
        loc, type_.getShape(), type_.getElementType()).getResult();
    isInitialized_ = true;
  }

  // Constructor with data initialization
  Tensor(const mlir::ArrayRef<int64_t> &shape, std::shared_ptr<T> data)
      : shapeStorage_(shape.begin(), shape.end()), shape_(shapeStorage_),
        src_(nullptr),
        type_(mlir::RankedTensorType::get(
            shape_,
            tensor_detail::cToMLIRType(
                vkml::Compiler::getInstance()->getContext(), typeid(T)))),
        data_(data), tensorValue_(nullptr), isInitialized_(false) {

    static int id_counter = 0;
    symbolic_id_ = "tensor_" + std::to_string(id_counter++);
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();

    auto scope = compiler->inMainBeforeTerminator();
    
    // Use arith.constant if data is provided, otherwise tensor.empty
    if (data_ != nullptr) {
      // Calculate total number of elements
      int64_t numElements = 1;
      for (auto dim : shape) {
        numElements *= dim;
      }
      
      // Create DenseElementsAttr from the raw data
      auto elemType = type_.getElementType();
      mlir::Attribute initialValueAttr;
      
      if constexpr (std::is_same_v<T, float>) {
        if (elemType.isF32()) {
          llvm::ArrayRef<float> dataArray(data_.get(), numElements);
          initialValueAttr = mlir::DenseElementsAttr::get(type_, dataArray);
        }
      } else if constexpr (std::is_same_v<T, double>) {
        if (elemType.isF64()) {
          llvm::ArrayRef<double> dataArray(data_.get(), numElements);
          initialValueAttr = mlir::DenseElementsAttr::get(type_, dataArray);
        }
      } else if constexpr (std::is_integral_v<T>) {
        llvm::ArrayRef<T> dataArray(data_.get(), numElements);
        initialValueAttr = mlir::DenseElementsAttr::get(type_, dataArray);
      }
      
      if (initialValueAttr) {
        tensorValue_ = builder.create<mlir::arith::ConstantOp>(
            loc, type_, initialValueAttr).getResult();
      } else {
        tensorValue_ = builder.create<mlir::tensor::EmptyOp>(
            loc, type_.getShape(), type_.getElementType()).getResult();
      }
    } else {
      tensorValue_ = builder.create<mlir::tensor::EmptyOp>(
          loc, type_.getShape(), type_.getElementType()).getResult();
    }
    isInitialized_ = true;
  }

  // Helper method to set data after construction
  void setData(std::shared_ptr<T> data) {
    data_ = data;
    // Note: This won't update the MLIR variable's initial value
    // You'd need to use tosa.variable_write to update it at runtime
  }

  // Helper method to get data
  std::shared_ptr<T> getData() const { return data_; }

  explicit Tensor(const T &scalar, const mlir::ArrayRef<int64_t> &shape)
      : Tensor<T>(shape) {
    // Initialize mlir::Value to represent the scalar
  }

private:
  inline mlir::Value read() const {
    if (!isInitialized_ || !tensorValue_) {
      throw std::runtime_error("Tensor not initialized");
    }
    return tensorValue_;
  }

  inline void write(mlir::Value newValue) {
    tensorValue_ = newValue;
    isInitialized_ = true;
  }

  template <bool isIncrement> void applyInPlaceIncrementDecrement() {
    static_assert(
        std::is_arithmetic_v<T>,
        "Increment/decrement only supported for arithmetic tensor types");
    auto &builder = vkml::Compiler::getInstance()->getBuilder();
    auto loc = builder.getUnknownLoc();
    auto ctx = vkml::Compiler::getInstance()->getContext();

    mlir::Type elemTy = tensor_detail::cToMLIRType(ctx, typeid(T));
    // Use shape {1} to ease creating DenseElementsAttr uniformly.
    mlir::RankedTensorType oneTy = mlir::RankedTensorType::get({1}, elemTy);
    mlir::Attribute elementAttr;
    if constexpr (std::is_floating_point_v<T>) {
      if (elemTy.isF32())
        elementAttr = builder.getF32FloatAttr(1.0f);
      else if (elemTy.isF64())
        elementAttr = builder.getF64FloatAttr(1.0);
      else
        elementAttr = builder.getF32FloatAttr(1.0f);
    } else {
      elementAttr = builder.getIntegerAttr(elemTy, 1);
    }
    std::array<mlir::Attribute, 1> attrArr{elementAttr};
    auto valueAttr = mlir::DenseElementsAttr::get(
        oneTy, llvm::ArrayRef<mlir::Attribute>(attrArr));
    auto constOp = builder.create<mlir::arith::ConstantOp>(loc, oneTy, valueAttr);
    Tensor<T> oneTensor({1});
    oneTensor.write(constOp.getResult());

    // Use existing arithmetic helpers (broadcast will occur via TOSA rules)
    if constexpr (isIncrement) {
      auto updated = (*this) + oneTensor; // returns new tensor
      this->write(updated.read().getResult());
    } else {
      auto updated = (*this) - oneTensor;
      this->write(updated.read().getResult());
    }
  }

public:
  // Expose element type for template utilities (e.g., buildFunctionWrapper)
  template <typename Op, typename ReturnType, typename... Args>
  static auto buildFunctionWrapper(Args &&...args) {
    // Remove references and cv-qualifiers, then map Tensor<E> -> E and compute
    // common type.

    auto compiler = vkml::Compiler::getInstance();
    auto scope = compiler->inModuleStart();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    auto ctx = compiler->getContext();
    llvm::SmallVector<mlir::Type> argTypes;
    (argTypes.push_back(std::forward<Args>(args).read().getType()),
     ...);
    llvm::SmallVector<mlir::Value> argValues;
    (argValues.push_back(std::forward<Args>(args).read()), ...);
    llvm::SmallVector<mlir::ShapedTypeComponents> inferred;

    if (mlir::failed(Op::inferReturnTypeComponents(
            ctx, std::optional<mlir::Location>{loc},
            mlir::ValueRange{argValues},
            /*attrs=*/mlir::DictionaryAttr{},
            /*properties=*/nullptr,
            /*regions=*/{}, inferred))) {
      throw std::runtime_error("Op::inferReturnTypes failed");
    }
    auto elementType = tensor_detail::cToMLIRType(ctx, typeid(ReturnType));
    auto resultType =
        mlir::RankedTensorType::get(inferred[0].getDims(), elementType);

    auto func = compiler->createFunctionOp(Op::getOperationName().str(), argValues, {resultType}, [&](
        mlir::OpBuilder &builder, mlir::func::FuncOp func,
        mlir::Block &block) {
        auto op = builder.create<Op>(loc, resultType, func.getArguments());
    });

    auto mainScope = compiler->inMainBeforeTerminator();
    auto callOp = builder.create<mlir::func::CallOp>(loc, func, argValues);

    auto output = Tensor<ReturnType>(resultType.getShape());
    output.write(callOp.getResult(0));
    return std::move(output);
  }

  const std::vector<int64_t> &getShape() const { return shapeStorage_; }
  std::string getSymbolicId() const { return symbolic_id_; }

  // Conversion constructor: Tensor<U> from Tensor<T>
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  explicit Tensor(const Tensor<U> &other) {
    auto readValue = other.read();
    auto &builder = vkml::Compiler::getInstance()->getBuilder();
    auto loc = builder.getUnknownLoc();
    auto ctx = vkml::Compiler::getInstance()->getContext();
    
    auto elementType = tensor_detail::cToMLIRType(ctx, typeid(T));
    auto inputType = readValue.getType().cast<mlir::RankedTensorType>();
    auto resultType = mlir::RankedTensorType::get(inputType.getShape(), elementType);
    
    // Create linalg.generic for cast operation
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);
    
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        builder.getMultiDimIdentityMap(inputType.getRank()),
        builder.getMultiDimIdentityMap(inputType.getRank())
    };
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        inputType.getRank(), mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, resultType, readValue, emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          mlir::Value casted;
          if (elementType.isF32() || elementType.isF64()) {
            casted = b.create<mlir::arith::SIToFPOp>(nestedLoc, elementType, args[0]);
          } else if (elementType.isInteger(32) || elementType.isInteger(64)) {
            casted = b.create<mlir::arith::FPToSIOp>(nestedLoc, elementType, args[0]);
          } else {
            casted = args[0]; // Same type, no conversion needed
          }
          b.create<mlir::linalg::YieldOp>(nestedLoc, casted);
        });
    
    this->write(genericOp.getResult(0));
    this->shapeStorage_ = other.getShape();
    this->shape_ = shapeStorage_;
    this->type_ = resultType;
    this->symbolic_id_ = other.getSymbolicId() + "_casted";
  }

  // Helper to create linalg.generic for binary operations
  template<typename ArithOp, typename U, typename V>
  static auto linalgBinaryOp(const Tensor<U> &lhs, const Tensor<V> &rhs) {
    using ResultType = std::common_type_t<U, V>;
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    auto ctx = compiler->getContext();
    
    auto lhsValue = lhs.read();
    auto rhsValue = rhs.read();
    auto lhsType = lhsValue.getType().cast<mlir::RankedTensorType>();
    auto rhsType = rhsValue.getType().cast<mlir::RankedTensorType>();
    
    auto elementType = tensor_detail::cToMLIRType(ctx, typeid(ResultType));
    
    // Determine output shape (broadcast semantics)
    llvm::SmallVector<int64_t> outputShape;
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    int64_t maxRank = std::max(lhsRank, rhsRank);
    
    for (int64_t i = 0; i < maxRank; ++i) {
      int64_t lhsDim = (i < lhsRank) ? lhsType.getShape()[lhsRank - 1 - i] : 1;
      int64_t rhsDim = (i < rhsRank) ? rhsType.getShape()[rhsRank - 1 - i] : 1;
      outputShape.push_back(std::max(lhsDim, rhsDim));
    }
    std::reverse(outputShape.begin(), outputShape.end());
    
    auto resultType = mlir::RankedTensorType::get(outputShape, elementType);
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, outputShape, elementType);
    
    // Create indexing maps for broadcasting
    llvm::SmallVector<mlir::AffineMap> indexingMaps;
    indexingMaps.push_back(builder.getMultiDimIdentityMap(maxRank));
    indexingMaps.push_back(builder.getMultiDimIdentityMap(maxRank));
    indexingMaps.push_back(builder.getMultiDimIdentityMap(maxRank));
    
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        maxRank, mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, resultType, mlir::ValueRange{lhsValue, rhsValue}, 
        emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          auto result = b.create<ArithOp>(nestedLoc, args[0], args[1]);
          b.create<mlir::linalg::YieldOp>(nestedLoc, result.getResult());
        });
    
    Tensor<ResultType> result(outputShape);
    result.write(genericOp.getResult(0));
    return result;
  }

  // Helper for unary operations
  template<typename ArithOp, typename V>
  static auto linalgUnaryOp(const Tensor<V> &input) {
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    
    auto inputValue = input.read();
    auto inputType = inputValue.getType().cast<mlir::RankedTensorType>();
    auto resultType = inputType;
    
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());
    
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        builder.getMultiDimIdentityMap(inputType.getRank()),
        builder.getMultiDimIdentityMap(inputType.getRank())
    };
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        inputType.getRank(), mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, resultType, inputValue, emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          auto result = b.create<ArithOp>(nestedLoc, args[0]);
          b.create<mlir::linalg::YieldOp>(nestedLoc, result.getResult());
        });
    
    Tensor<V> result(input.getShape());
    result.write(genericOp.getResult(0));
    return result;
  }

  template <typename Op, typename U, typename V>
  static auto logicalOpHelper(const Tensor<U> &lhs, const Tensor<V> &rhs) {
    return buildFunctionWrapper<Op, bool>(lhs, rhs);
  }

  template <typename Op, typename U, typename V>
  static auto binaryOpHelper(const Tensor<U> &lhs, const Tensor<V> &rhs) {
    return buildFunctionWrapper<Op, std::common_type_t<U, V>>(lhs, rhs);
  }

  template <typename Op, typename U, typename V, typename W>
  static auto ternaryOpHelper(const Tensor<U> &a, const Tensor<V> &b,
                              const Tensor<W> &c) {
    return buildFunctionWrapper<Op, std::common_type_t<U, V, W>>(a, b, c);
  }

  template <typename Op, typename V>
  static auto unaryOpHelper(const Tensor<V> &tensor) {
    return buildFunctionWrapper<Op, V>(tensor);
  }

public:
  // Binary arithmetic/logical operators (single template each)
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator+(const Tensor<U> &rhs) const {
    if constexpr (std::is_integral_v<U> && std::is_integral_v<T>) {
      return linalgBinaryOp<mlir::arith::AddIOp, T, U>(*this, rhs);
    } else {
      return linalgBinaryOp<mlir::arith::AddFOp, T, U>(*this, rhs);
    }
  }

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator-(const Tensor<U> &rhs) const {
    if constexpr (std::is_integral_v<U> && std::is_integral_v<T>) {
      return linalgBinaryOp<mlir::arith::SubIOp, T, U>(*this, rhs);
    } else {
      return linalgBinaryOp<mlir::arith::SubFOp, T, U>(*this, rhs);
    }
  }

  // Unified division operator: integer -> DivSIOp, floating -> DivFOp
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator/(const Tensor<U> &rhs) const {
    if constexpr (std::is_integral_v<U> && std::is_integral_v<T>) {
      return linalgBinaryOp<mlir::arith::DivSIOp, T, U>(*this, rhs);
    } else if constexpr (std::is_floating_point_v<U> &&
                         std::is_floating_point_v<T>) {
      return linalgBinaryOp<mlir::arith::DivFOp, T, U>(*this, rhs);
    } else {
      static_assert(std::is_same_v<U, void>,
                    "Mixed integral/floating division not supported");
    }
  }

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator*(const Tensor<U> &rhs) const {
    if constexpr (std::is_integral_v<U> && std::is_integral_v<T>) {
      return linalgBinaryOp<mlir::arith::MulIOp, T, U>(*this, rhs);
    } else {
      return linalgBinaryOp<mlir::arith::MulFOp, T, U>(*this, rhs);
    }
  }

  Tensor<T> operator+() const {
    // Unary plus returns absolute value
    if constexpr (std::is_floating_point_v<T>) {
      return linalgUnaryOp<mlir::math::AbsFOp, T>(*this);
    } else {
      return linalgUnaryOp<mlir::math::AbsIOp, T>(*this);
    }
  }
  
  Tensor<T> operator~() const {
    // Bitwise not: XOR with all 1s using linalg.generic
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    
    auto inputValue = this->read();
    auto inputType = inputValue.getType().cast<mlir::RankedTensorType>();
    auto elemType = inputType.getElementType();
    
    // Create a constant tensor of all 1s (all bits set)
    int64_t allOnes = -1;
    auto onesAttr = mlir::DenseElementsAttr::get(inputType, 
        builder.getIntegerAttr(elemType, allOnes));
    auto onesConstant = builder.create<mlir::arith::ConstantOp>(
        loc, inputType, onesAttr);
    
    // Use linalg.generic to perform XOR
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, inputType.getShape(), elemType);
    
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        builder.getMultiDimIdentityMap(inputType.getRank()),
        builder.getMultiDimIdentityMap(inputType.getRank()),
        builder.getMultiDimIdentityMap(inputType.getRank())
    };
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        inputType.getRank(), mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, inputType, mlir::ValueRange{inputValue, onesConstant.getResult()},
        emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          auto xorResult = b.create<mlir::arith::XOrIOp>(nestedLoc, args[0], args[1]);
          b.create<mlir::linalg::YieldOp>(nestedLoc, xorResult.getResult());
        });
    
    Tensor<T> result(this->getShape());
    result.write(genericOp.getResult(0));
    return result;
  }
  
  Tensor<T> operator!() const {
    // Logical not for boolean/integer types
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    
    auto inputValue = this->read();
    auto inputType = inputValue.getType().cast<mlir::RankedTensorType>();
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());
    
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        builder.getMultiDimIdentityMap(inputType.getRank()),
        builder.getMultiDimIdentityMap(inputType.getRank())
    };
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        inputType.getRank(), mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, inputType, inputValue, emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          auto zero = b.create<mlir::arith::ConstantIntOp>(nestedLoc, 0, 
              inputType.getElementType().getIntOrFloatBitWidth());
          auto cmp = b.create<mlir::arith::CmpIOp>(nestedLoc, 
              mlir::arith::CmpIPredicate::eq, args[0], zero);
          auto result = b.create<mlir::arith::ExtUIOp>(nestedLoc, 
              inputType.getElementType(), cmp);
          b.create<mlir::linalg::YieldOp>(nestedLoc, result.getResult());
        });
    
    Tensor<T> result(this->getShape());
    result.write(genericOp.getResult(0));
    return result;
  }

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<T> &&
                                                    std::is_integral_v<U>>>
  auto operator%(const Tensor<U> &rhs) const {
    // Modulo: a % b = a - (a / b) * b  (integer arithmetic semantics)
    // Reuse existing operator overloads to build IR safely.
    return *this - ((*this / rhs) * rhs);
  }

  template <typename U, typename = std::enable_if_t<
                            std::is_integral_v<U> && std::is_integral_v<T> &&
                            std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
  auto operator&(const Tensor<U> &rhs) const {
    return linalgBinaryOp<mlir::arith::AndIOp, T, U>(*this, rhs);
  }
  template <typename U, typename = std::enable_if_t<
                            std::is_integral_v<U> && std::is_integral_v<T> &&
                            std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
  auto operator|(const Tensor<U> &rhs) const {
    return linalgBinaryOp<mlir::arith::OrIOp, T, U>(*this, rhs);
  }
  template <typename U, typename = std::enable_if_t<
                            std::is_integral_v<U> && std::is_integral_v<T> &&
                            std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
  auto operator^(const Tensor<U> &rhs) const {
    return linalgBinaryOp<mlir::arith::XOrIOp, T, U>(*this, rhs);
  }

  template <typename U, typename = std::enable_if_t<
                            std::is_integral_v<U> && std::is_integral_v<T> &&
                            std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
  auto operator<<(const Tensor<U> &rhs) const {
    return linalgBinaryOp<mlir::arith::ShLIOp, T, U>(*this, rhs);
  }
  template <typename U, typename = std::enable_if_t<
                            std::is_integral_v<U> && std::is_integral_v<T> &&
                            std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
  auto operator>>(const Tensor<U> &rhs) const {
    return linalgBinaryOp<mlir::arith::ShRUIOp, T, U>(*this, rhs);
  }

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator&&(const Tensor<U> &rhs) const {
    return linalgBinaryOp<mlir::arith::AndIOp, T, U>(*this, rhs);
  }
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator||(const Tensor<U> &rhs) const {
    return linalgBinaryOp<mlir::arith::OrIOp, T, U>(*this, rhs);
  }

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator==(const Tensor<U> &rhs) const {
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    
    auto lhsValue = this->read();
    auto rhsValue = rhs.read();
    auto lhsType = lhsValue.getType().cast<mlir::RankedTensorType>();
    auto rhsType = rhsValue.getType().cast<mlir::RankedTensorType>();
    
    // Result is boolean tensor
    auto boolType = builder.getI1Type();
    auto resultType = mlir::RankedTensorType::get(lhsType.getShape(), boolType);
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, lhsType.getShape(), boolType);
    
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        builder.getMultiDimIdentityMap(lhsType.getRank()),
        builder.getMultiDimIdentityMap(lhsType.getRank()),
        builder.getMultiDimIdentityMap(lhsType.getRank())
    };
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        lhsType.getRank(), mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, resultType, mlir::ValueRange{lhsValue, rhsValue}, 
        emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          mlir::Value cmp;
          if (lhsType.getElementType().isIntOrIndex()) {
            cmp = b.create<mlir::arith::CmpIOp>(nestedLoc, 
                mlir::arith::CmpIPredicate::eq, args[0], args[1]);
          } else {
            cmp = b.create<mlir::arith::CmpFOp>(nestedLoc, 
                mlir::arith::CmpFPredicate::OEQ, args[0], args[1]);
          }
          b.create<mlir::linalg::YieldOp>(nestedLoc, cmp);
        });
    
    Tensor<bool> result(std::vector<int64_t>(lhsType.getShape().begin(), 
                                             lhsType.getShape().end()));
    result.write(genericOp.getResult(0));
    return result;
  }
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator!=(const Tensor<U> &rhs) const {
    return !(*this == rhs);
  }
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator>(const Tensor<U> &rhs) const {
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    
    auto lhsValue = this->read();
    auto rhsValue = rhs.read();
    auto lhsType = lhsValue.getType().cast<mlir::RankedTensorType>();
    
    auto boolType = builder.getI1Type();
    auto resultType = mlir::RankedTensorType::get(lhsType.getShape(), boolType);
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, lhsType.getShape(), boolType);
    
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        builder.getMultiDimIdentityMap(lhsType.getRank()),
        builder.getMultiDimIdentityMap(lhsType.getRank()),
        builder.getMultiDimIdentityMap(lhsType.getRank())
    };
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        lhsType.getRank(), mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, resultType, mlir::ValueRange{lhsValue, rhsValue}, 
        emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          mlir::Value cmp;
          if (lhsType.getElementType().isIntOrIndex()) {
            cmp = b.create<mlir::arith::CmpIOp>(nestedLoc, 
                mlir::arith::CmpIPredicate::sgt, args[0], args[1]);
          } else {
            cmp = b.create<mlir::arith::CmpFOp>(nestedLoc, 
                mlir::arith::CmpFPredicate::OGT, args[0], args[1]);
          }
          b.create<mlir::linalg::YieldOp>(nestedLoc, cmp);
        });
    
    Tensor<bool> result(std::vector<int64_t>(lhsType.getShape().begin(), 
                                             lhsType.getShape().end()));
    result.write(genericOp.getResult(0));
    return result;
  }
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator>=(const Tensor<U> &rhs) const {
    auto compiler = vkml::Compiler::getInstance();
    auto &builder = compiler->getBuilder();
    auto loc = builder.getUnknownLoc();
    
    auto lhsValue = this->read();
    auto rhsValue = rhs.read();
    auto lhsType = lhsValue.getType().cast<mlir::RankedTensorType>();
    
    auto boolType = builder.getI1Type();
    auto resultType = mlir::RankedTensorType::get(lhsType.getShape(), boolType);
    auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
        loc, lhsType.getShape(), boolType);
    
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {
        builder.getMultiDimIdentityMap(lhsType.getRank()),
        builder.getMultiDimIdentityMap(lhsType.getRank()),
        builder.getMultiDimIdentityMap(lhsType.getRank())
    };
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        lhsType.getRank(), mlir::utils::IteratorType::parallel);
    
    auto genericOp = builder.create<mlir::linalg::GenericOp>(
        loc, resultType, mlir::ValueRange{lhsValue, rhsValue}, 
        emptyTensor.getResult(),
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::ValueRange args) {
          mlir::Value cmp;
          if (lhsType.getElementType().isIntOrIndex()) {
            cmp = b.create<mlir::arith::CmpIOp>(nestedLoc, 
                mlir::arith::CmpIPredicate::sge, args[0], args[1]);
          } else {
            cmp = b.create<mlir::arith::CmpFOp>(nestedLoc, 
                mlir::arith::CmpFPredicate::OGE, args[0], args[1]);
          }
          b.create<mlir::linalg::YieldOp>(nestedLoc, cmp);
        });
    
    Tensor<bool> result(std::vector<int64_t>(lhsType.getShape().begin(), 
                                             lhsType.getShape().end()));
    result.write(genericOp.getResult(0));
    return result;
  }

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator<(const Tensor<U> &rhs) const {
    return !(*this >= rhs);
  }
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                                    std::is_arithmetic_v<T>>>
  auto operator<=(const Tensor<U> &rhs) const {
    return !(*this > rhs);
  }

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
    // Print a readable representation using the symbolic id and shape.
    os << t.symbolic_id_ << "(";
    for (std::size_t i = 0; i < t.shape_.size(); ++i) {
      if (i)
        os << "x";
      os << t.shape_[i];
    }
    os << ")";

    // Print the MLIR type.
    std::string mlir_str;
    llvm::raw_string_ostream llvm_os(mlir_str);
    t.type_.print(llvm_os);
    llvm_os.flush();
    os << " " << mlir_str;
    return os;
  }

  Tensor<T> &operator++() {
    applyInPlaceIncrementDecrement</*isIncrement=*/true>();
    return *this;
  }

  Tensor<T> &operator--() {
    applyInPlaceIncrementDecrement</*isIncrement=*/false>();
    return *this;
  }

  // Postfix increment
  Tensor<T> operator++(int) {
    applyInPlaceIncrementDecrement</*isIncrement=*/true>();
    return *this;
  }

  // Postfix decrement
  Tensor<T> operator--(int) {
    applyInPlaceIncrementDecrement</*isIncrement=*/false>();
    return *this;
  }

  // Subscript operator
  // Option A implementation: return a new Tensor<T> representing a slice along
  // the first dimension. Currently only supports a single integral index into
  // the first dimension; rank is reduced by 1. If original shape is [D0, D1,
  // ..., Dn] result shape is [D1, ..., Dn].
  template <typename IndexType,
            typename = std::enable_if_t<std::is_integral_v<IndexType> &&
                                        std::is_unsigned_v<IndexType>>>
  Tensor<T> operator[](IndexType index) const {
    if (shape_.empty()) {
      throw std::out_of_range("Cannot index into a rank-0 tensor");
    }
    if (index >= static_cast<IndexType>(shape_[0])) {
      throw std::out_of_range("Index out of bounds for first dimension");
    }
    auto &builder = vkml::Compiler::getInstance()->getBuilder();
    auto loc = builder.getUnknownLoc();
    auto ctx = vkml::Compiler::getInstance()->getContext();

    // Build offsets, sizes, and strides for tensor.extract_slice
    const int rank = static_cast<int>(shape_.size());
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    llvm::SmallVector<mlir::OpFoldResult> strides;
    
    // First dimension: extract at index
    offsets.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(index)));
    sizes.push_back(builder.getI64IntegerAttr(1));
    strides.push_back(builder.getI64IntegerAttr(1));
    
    // Remaining dimensions: full range
    for (int i = 1; i < rank; ++i) {
      offsets.push_back(builder.getI64IntegerAttr(0));
      sizes.push_back(builder.getI64IntegerAttr(shape_[i]));
      strides.push_back(builder.getI64IntegerAttr(1));
    }
    
    auto inputValue = this->read();
    auto inputType = inputValue.getType().cast<mlir::RankedTensorType>();
    auto elementType = inputType.getElementType();
    
    // Result shape with size 1 in first dimension
    llvm::SmallVector<int64_t> slicedShape(shape_.begin(), shape_.end());
    slicedShape[0] = 1;
    auto slicedType = mlir::RankedTensorType::get(slicedShape, elementType);
    
    auto extractSliceOp = builder.create<mlir::tensor::ExtractSliceOp>(
        loc, slicedType, inputValue, offsets, sizes, strides);

    // Build reduced-rank shape for API (drop first dim)
    std::vector<int64_t> reducedShape;
    reducedShape.reserve(shape_.size() - 1);
    for (size_t i = 1; i < shape_.size(); ++i)
      reducedShape.push_back(shape_[i]);
    Tensor<T> result(reducedShape);
    result.write(extractSliceOp.getResult());
    return result;
  }

  // Overload taking a Tensor index (must be scalar tensor with compatible
  // unsigned integral element). Not yet implemented: will throw.
  template <typename IndexTensorType,
            typename = std::enable_if_t<std::is_integral_v<IndexTensorType> &&
                                        std::is_unsigned_v<IndexTensorType>>>
  Tensor<T> operator[](const Tensor<IndexTensorType> & /*indexTensor*/) const {
    throw std::runtime_error(
        "Tensor index operand not yet supported for operator[]");
  }

  // // Function call operator
  // template<typename U>
  // friend Tensor<std::common_type_t<U, T>> operator,(const Tensor<U>& lhs,
  // const Tensor<T>& rhs) { return Tensor<std::common_type_t<U,
  // T>>(tensor_detail::broadcastShapes(lhs.getShape(), rhs.getShape())); }

  // Assignment operator
  Tensor<T> &operator=(const Tensor &rhs) = delete;

private:
};