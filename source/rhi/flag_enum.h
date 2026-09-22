/**
 * {file}
 * {brief} Marks an enumeration whose enumerators are bit flags.
 *
 * A combination of flags is a value the type is meant to hold, not an out-of-range cast. Clang
 * understands that when told, which is what lets its analyzer tell a real out-of-range value from
 * BufferUsage::Storage | BufferUsage::DeviceAddress. Other compilers are told nothing.
 */
#ifndef LIGHT_RHI_FLAG_ENUM_H
#define LIGHT_RHI_FLAG_ENUM_H

#ifdef __clang__
#define LIGHT_RHI_FLAG_ENUM [[clang::flag_enum]]
#else
#define LIGHT_RHI_FLAG_ENUM
#endif

#endif // LIGHT_RHI_FLAG_ENUM_H
