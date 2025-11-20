#include <cstdint>
#include <stdfloat>
#include <stdint.h>
#include <hash_fun.h>

#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace vkml {
    struct entity 
    {        
        mlir::RankedTensorType type_;
    };

    struct entity_hash {
        uint64_t operator()(const entity& e) const noexcept {
            return 0;
        }
    };

    template<int64_t... I1s, int64_t... I2s>
    void broad_cast_helper(std::index_sequence<I1s...>, std::index_sequence<I2s...>) {
        // dummy

    }

    template<typename T>
    class tensor {
        entity entity_;
        mlir::OpBuilder& builder_;
        
        mlir::Type setElementType() const {
            return mlir::Type();
        }
    
    public:
        ~tensor() = default;
    
        tensor(const mlir::ArrayRef<int64_t>& shape){
            entity_.type_ = mlir::RankedTensorType::get(shape, setElementType());
        }                
    
        template<typename U>
        operator tensor<U>() const {
            // cast to U
            return tensor<U>(entity_.type_.getShape());
        }

        tensor(const tensor<T> & other) {
            entity_.type_ = other.entity_.type_;
        }

        // operators a=b, a+=b, a-=b, a*=b, a/=b, a%=b, 
        template<typename T2>
        tensor<T>& operator=(const tensor<T2> &rhs) const {
            return static_cast<tensor<T>>(rhs.entity_.type_.getShape()); // dummy
        }

        template<typename T2>
        tensor<T>& operator+=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator+(const tensor<T2> &rhs) const {
            this += rhs;
            return *this;
        }
        
        template<typename T2>
        tensor<T>& operator-=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator-(const tensor<T2> &rhs) const {
            this -= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator*=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator*(const tensor<T2> &rhs) const {
            this *= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator/=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator/(const tensor<T2> &rhs) const {
            this /= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator%=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator%(const tensor<T2> &rhs) const {
            this %= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator&=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator&(const tensor<T2> &rhs) {
            this &= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator|=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator|(const tensor<T2> &rhs) {
            this |= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator^=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator^(const tensor<T2> &rhs) {
            this ^= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator<<=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator<<(const tensor<T2> &rhs) {
            this <<= rhs;
            return *this;
        }

        template<typename T2>
        tensor<T>& operator>>=(const tensor<T2> &rhs) {
            return *this;
        }

        template<typename T2>
        tensor<T>& operator>>(const tensor<T2> &rhs) {
            this >>= rhs;
            return *this;
        }

        // operators ++a, a++, --a, a--
        tensor<T>& operator++() {
            return *this;
        }

        tensor<T>& operator--() {
            return *this;
        }

        // operators +a, -a, ~a,
        tensor<T> operator+() const {
            return *this;
        }

        tensor<T> operator-() const {
            return *this;
        }

        tensor<T> operator~() const {
            return *this;
        }

        // operators !a, a && b, a || b
        tensor<bool> operator!() const {
            return tensor<bool>(entity_.type_.getShape());
        }
        template<typename T2>
        tensor<bool> operator&&(const tensor<T2> &rhs) const {
            return tensor<bool>(entity_.type_.getShape());
        }
        template<typename T2>
        tensor<bool> operator||(const tensor<T2> &rhs) const {
            return tensor<bool>(entity_.type_.getShape());
        }

        // operators a == b,  a < b,
        template<typename T2>
        tensor<bool> operator==(const tensor<T2> &rhs) const {
            return tensor<bool>(entity_.type_.getShape());
        }

        template<typename T2>
        tensor<bool> operator<(const tensor<T2> &rhs) const {
            return tensor<bool>(entity_.type_.getShape());
        }

        // operators a[...], *a, &a, a->b, a->*b

        // operators a(...), (a, b)
        
    };
        
    template<>
    inline mlir::Type tensor<float>::setElementType() const {
        return builder_.getF32Type();
    }
    template<>
    inline mlir::Type tensor<double>::setElementType() const {
        return builder_.getF64Type();
    }
    template<>
    inline mlir::Type tensor<uint8_t>::setElementType() const {
        return builder_.getIntegerType(8, mlir::IntegerType::Unsigned);
    }
    template<>
    inline mlir::Type tensor<int8_t>::setElementType() const {
        return builder_.getIntegerType(8, mlir::IntegerType::Signed);
    }
    template<>
    inline mlir::Type tensor<uint16_t>::setElementType() const {
        return builder_.getIntegerType(16, mlir::IntegerType::Unsigned);
    }
    template<>
    inline mlir::Type tensor<int16_t>::setElementType() const {
        return builder_.getIntegerType(16, mlir::IntegerType::Signed);
    }
    template<>
    inline mlir::Type tensor<uint32_t>::setElementType() const {
        return builder_.getIntegerType(32, mlir::IntegerType::Unsigned);
    } 
    template<>
    inline mlir::Type tensor<int32_t>::setElementType() const {
        return builder_.getIntegerType(32, mlir::IntegerType::Signed);
    }
    template<>
    inline mlir::Type tensor<uint64_t>::setElementType() const {
        return builder_.getIntegerType(64, mlir::IntegerType::Unsigned);
    }
    template<>
    inline mlir::Type tensor<int64_t>::setElementType() const {
        return builder_.getIntegerType(64, mlir::IntegerType::Signed);
    }

#if defined(__STDCPP_FLOAT16_T__)
    template<>
    inline mlir::Type tensor<std::float16_t>::setElementType() const {
        return builder_.getF16Type();
    }
#endif

#if defined (__STDCPP_BFLOAT16_T__) 
    template<>
    inline mlir::Type tensor<std::bfloat16_t>::setElementType() const {
        return builder_.getBF16Type();
    }
#endif

#if defined(__STDCPP_FLOAT128_T__)
    template<>
    inline mlir::Type tensor<std::float128_t>::setElementType() const {
        return builder_.getF128Type();
    }
#endif 

}