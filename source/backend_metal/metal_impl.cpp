// metal_impl.cpp
// Exactly ONE translation unit must define these macros before including
// the metal-cpp headers. They instantiate the function bodies.
// All other .cpp files in this backend include the headers WITHOUT these macros.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
