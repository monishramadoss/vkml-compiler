  ✓ Default pipeline completed
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module {
  func.func @main() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32>
    %0 = call @arith.addf_linalg(%alloc, %alloc_0) : (memref<2x3xf32>, memref<1x3xf32>) -> memref<2x3xf32>
    %1 = call @arith.subf_linalg(%0, %alloc) : (memref<2x3xf32>, memref<2x3xf32>) -> memref<2x3xf32>
    return
  }
  func.func @arith.addf_linalg(%arg0: memref<2x3xf32>, %arg1: memref<1x3xf32>) -> memref<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%c2)[%c0, %c1]
    %1 = affine.apply #map(%c3)[%c0, %c1]
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %1, %arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_0, %arg12 = %c1_0, %arg13 = %c1_0) {
      %2 = affine.apply #map1(%arg2)[%c1, %c0]
      %3 = affine.apply #map1(%arg3)[%c1, %c0]
      %4 = memref.load %arg0[%2, %3] : memref<2x3xf32>
      %5 = memref.load %arg1[%c0, %3] : memref<1x3xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %alloc[%2, %3] : memref<2x3xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    return %alloc : memref<2x3xf32>
  }
  func.func @arith.subf_linalg(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) -> memref<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%c2)[%c0, %c1]
    %1 = affine.apply #map(%c3)[%c0, %c1]
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %1, %arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_0, %arg12 = %c1_0, %arg13 = %c1_0) {
      %2 = affine.apply #map1(%arg2)[%c1, %c0]
      %3 = affine.apply #map1(%arg3)[%c1, %c0]
      %4 = memref.load %arg0[%2, %3] : memref<2x3xf32>
      %5 = memref.load %arg1[%2, %3] : memref<2x3xf32>
      %6 = arith.subf %4, %5 : f32
      memref.store %6, %alloc[%2, %3] : memref<2x3xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    return %alloc : memref<2x3xf32>
  }
}


 Running outlining passes
  ✓ Outlining completed
module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, api=Vulkan, #spirv.resource_limits<>>} {
  func.func @main() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32>
    %0 = call @arith.addf_linalg(%alloc, %alloc_0) : (memref<2x3xf32>, memref<1x3xf32>) -> memref<2x3xf32>
    %1 = call @arith.subf_linalg(%0, %alloc) : (memref<2x3xf32>, memref<2x3xf32>) -> memref<2x3xf32>
    return
  }
  func.func @arith.addf_linalg(%arg0: memref<2x3xf32>, %arg1: memref<1x3xf32>) -> memref<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %c1_0 = arith.constant 1 : index
    %c2_1 = arith.constant 2 : index
    %c3_2 = arith.constant 3 : index
    gpu.launch_func  @arith.addf_linalg_kernel::@arith.addf_linalg_kernel blocks in (%c2_1, %c3_2, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32>, %arg1 : memref<1x3xf32>, %alloc : memref<2x3xf32>)
    return %alloc : memref<2x3xf32>
  }
  gpu.module @arith.addf_linalg_kernel {
    gpu.func @arith.addf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32>, %arg3: memref<1x3xf32>, %arg4: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32>
      %5 = memref.load %arg3[%arg1, %3] : memref<1x3xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32>
      gpu.return
    }
  }
  func.func @arith.subf_linalg(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) -> memref<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %c1_0 = arith.constant 1 : index
    %c2_1 = arith.constant 2 : index
    %c3_2 = arith.constant 3 : index
    gpu.launch_func  @arith.subf_linalg_kernel::@arith.subf_linalg_kernel blocks in (%c2_1, %c3_2, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32>, %arg1 : memref<2x3xf32>, %alloc : memref<2x3xf32>)
    return %alloc : memref<2x3xf32>
  }
  gpu.module @arith.subf_linalg_kernel {
    gpu.func @arith.subf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32>, %arg3: memref<2x3xf32>, %arg4: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32>
      %5 = memref.load %arg3[%1, %3] : memref<2x3xf32>
      %6 = arith.subf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32>
      gpu.return
    }
  }
}

  ✓ Entry point ABI set on gpu.func ops
module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, api=Vulkan, #spirv.resource_limits<>>} {
  func.func @main() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32>
    %0 = call @arith.addf_linalg(%alloc, %alloc_0) : (memref<2x3xf32>, memref<1x3xf32>) -> memref<2x3xf32>
    %1 = call @arith.subf_linalg(%0, %alloc) : (memref<2x3xf32>, memref<2x3xf32>) -> memref<2x3xf32>
    return
  }
  func.func @arith.addf_linalg(%arg0: memref<2x3xf32>, %arg1: memref<1x3xf32>) -> memref<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %c1_0 = arith.constant 1 : index
    %c2_1 = arith.constant 2 : index
    %c3_2 = arith.constant 3 : index
    gpu.launch_func  @arith.addf_linalg_kernel::@arith.addf_linalg_kernel blocks in (%c2_1, %c3_2, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32>, %arg1 : memref<1x3xf32>, %alloc : memref<2x3xf32>)
    return %alloc : memref<2x3xf32>
  }
  gpu.module @arith.addf_linalg_kernel {
    gpu.func @arith.addf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32>, %arg3: memref<1x3xf32>, %arg4: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32>
      %5 = memref.load %arg3[%arg1, %3] : memref<1x3xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32>
      gpu.return
    }
  }
  func.func @arith.subf_linalg(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) -> memref<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %c1_0 = arith.constant 1 : index
    %c2_1 = arith.constant 2 : index
    %c3_2 = arith.constant 3 : index
    gpu.launch_func  @arith.subf_linalg_kernel::@arith.subf_linalg_kernel blocks in (%c2_1, %c3_2, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32>, %arg1 : memref<2x3xf32>, %alloc : memref<2x3xf32>)
    return %alloc : memref<2x3xf32>
  }
  gpu.module @arith.subf_linalg_kernel {
    gpu.func @arith.subf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32>, %arg3: memref<2x3xf32>, %arg4: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32>
      %5 = memref.load %arg3[%1, %3] : memref<2x3xf32>
      %6 = arith.subf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32>
      gpu.return
    }
  }
}


 Running SPIR-V conversion passes
  ✓ SPIR-V conversion completed
module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, api=Vulkan, #spirv.resource_limits<>>} {
  func.func @main() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32, #spirv.storage_class<StorageBuffer>>
    %0 = call @arith.addf_linalg(%alloc, %alloc_0) : (memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, memref<1x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    %1 = call @arith.subf_linalg(%0, %alloc) : (memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    return
  }
  func.func @arith.addf_linalg(%arg0: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<1x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    gpu.launch_func  @arith.addf_linalg_kernel::@arith.addf_linalg_kernel blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : memref<1x3xf32, #spirv.storage_class<StorageBuffer>>, %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>)
    return %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
  }
  spirv.module @__spv__arith.addf_linalg_kernel Logical GLSL450 {
    spirv.GlobalVariable @__builtin__NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.func @arith.addf_linalg_kernel(%arg0: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>}, %arg1: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1), StorageBuffer>}, %arg2: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}, %arg3: !spirv.ptr<!spirv.struct<(!spirv.array<3 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 3)>}, %arg4: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 4)>}) "None" attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>, workgroup_attributions = 0 : i64} {
      %cst3_i32 = spirv.Constant 3 : i32
      %cst0_i32 = spirv.Constant 0 : i32
      %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %0 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %2 = spirv.Load "Input" %__builtin__WorkgroupId___addr_0 : vector<3xi32>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_1 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %4 = spirv.Load "Input" %__builtin__WorkgroupId___addr_1 : vector<3xi32>
      %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %5 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
      %__builtin__LocalInvocationId___addr_2 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %6 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_2 : vector<3xi32>
      %__builtin__LocalInvocationId___addr_3 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %7 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_3 : vector<3xi32>
      %__builtin__NumWorkgroups___addr = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %8 = spirv.Load "Input" %__builtin__NumWorkgroups___addr : vector<3xi32>
      %__builtin__NumWorkgroups___addr_4 = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %9 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_4 : vector<3xi32>
      %__builtin__NumWorkgroups___addr_5 = spirv.mlir.addressof @__builtin__lNumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %10 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_5 : vector<3xi32>
      %11 = spirv.IMul %1, %arg0 : i32
      %12 = spirv.IAdd %11, %arg1 : i32
      %13 = spirv.IMul %3, %arg0 : i32
      %14 = spirv.IAdd %13, %arg1 : i32
      %15 = spirv.IMul %12, %cst3_i32 : i32
      %16 = spirv.IAdd %14, %15 : i32
      %17 = spirv.AccessChain %arg2[%cst0_i32, %16] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %18 = spirv.Load "StorageBuffer" %17 : f32
      %19 = spirv.IMul %arg1, %cst3_i32 : i32
      %20 = spirv.IAdd %14, %19 : i32
      %21 = spirv.AccessChain %arg3[%cst0_i32, %20] : !spirv.ptr<!spirv.struct<(!spirv.array<3 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %22 = spirv.Load "StorageBuffer" %21 : f32
      %23 = spirv.FAdd %18, %22 : f32
      %24 = spirv.IMul %12, %cst3_i32 : i32
      %25 = spirv.IAdd %14, %24 : i32
      %26 = spirv.AccessChain %arg4[%cst0_i32, %25] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %26, %23 : f32
      spirv.Return
    }
  }
  gpu.module @arith.addf_linalg_kernel {
    gpu.func @arith.addf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg3: memref<1x3xf32, #spirv.storage_class<StorageBuffer>>, %arg4: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      %5 = memref.load %arg3[%arg1, %3] : memref<1x3xf32, #spirv.storage_class<StorageBuffer>>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      gpu.return
    }
  }
  func.func @arith.subf_linalg(%arg0: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    gpu.launch_func  @arith.subf_linalg_kernel::@arith.subf_linalg_kernel blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>)
    return %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
  }
  spirv.module @__spv__arith.subf_linalg_kernel Logical GLSL450 {
    spirv.GlobalVariable @__builtin__NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.func @arith.subf_linalg_kernel(%arg0: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>}, %arg1: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1), StorageBuffer>}, %arg2: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}, %arg3: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 3)>}, %arg4: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 4)>}) "None" attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>, workgroup_attributions = 0 : i64} {
      %cst3_i32 = spirv.Constant 3 : i32
      %cst0_i32 = spirv.Constant 0 : i32
      %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %0 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %2 = spirv.Load "Input" %__builtin__WorkgroupId___addr_0 : vector<3xi32>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_1 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %4 = spirv.Load "Input" %__builtin__WorkgroupId___addr_1 : vector<3xi32>
      %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %5 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
      %__builtin__LocalInvocationId___addr_2 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %6 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_2 : vector<3xi32>
      %__builtin__LocalInvocationId___addr_3 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %7 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_3 : vector<3xi32>
      %__builtin__NumWorkgroups___addr = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %8 = spirv.Load "Input" %__builtin__NumWorkgroups___addr : vector<3xi32>
      %__builtin__NumWorkgroups___addr_4 = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %9 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_4 : vector<3xi32>
      %__builtin__NumWorkgroups___addr_5 = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %10 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_5 : vector<3xi32>
      %11 = spirv.IMul %1, %arg0 : i32
      %12 = spirv.IAdd %11, %arg1 : i32
      %13 = spirv.IMul %3, %arg0 : i32
      %14 = spirv.IAdd %13, %arg1 : i32
      %15 = spirv.IMul %12, %cst3_i32 : i32
      %16 = spirv.IAdd %14, %15 : i32
      %17 = spirv.AccessChain %arg2[%cst0_i32, %16] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %18 = spirv.Load "StorageBuffer" %17 : f32
      %19 = spirv.IMul %12, %cst3_i32 : i32
      %20 = spirv.IAdd %14, %19 : i32
      %21 = spirv.AccessChain %arg3[%cst0_i32, %20] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %22 = spirv.Load "StorageBuffer" %21 : f32
      %23 = spirv.FSub %18, %22 : f32
      %24 = spirv.IMul %12, %cst3_i32 : i32
      %25 = spirv.IAdd %14, %24 : i32
      %26 = spirv.AccessChain %arg4[%cst0_i32, %25] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %26, %23 : f32
      spirv.Return
    }
  }
  gpu.module @arith.subf_linalg_kernel {
    gpu.func @arith.subf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg3: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg4: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      %5 = memref.load %arg3[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      %6 = arith.subf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      gpu.return
    }
  }
}

  ✓ GPU modules erased
module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, api=Vulkan, #spirv.resource_limits<>>} {
  func.func @main() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32, #spirv.storage_class<StorageBuffer>>
    %0 = call @arith.addf_linalg(%alloc, %alloc_0) : (memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, memref<1x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    %1 = call @arith.subf_linalg(%0, %alloc) : (memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    return
  }
  func.func @arith.addf_linalg(%arg0: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<1x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    gpu.launch_func  @arith.addf_linalg_kernel::@arith.addf_linalg_kernel blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : memref<1x3xf32, #spirv.storage_class<StorageBuffer>>, %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>)
    return %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
  }
  spirv.module @__spv__arith.addf_linalg_kernel Logical GLSL450 {
    spirv.GlobalVariable @__builtin__NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.func @arith.addf_linalg_kernel(%arg0: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>}, %arg1: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1), StorageBuffer>}, %arg2: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}, %arg3: !spirv.ptr<!spirv.struct<(!spirv.array<3 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 3)>}, %arg4: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 4)>}) "None" attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>, workgroup_attributions = 0 : i64} {
      %cst3_i32 = spirv.Constant 3 : i32
      %cst0_i32 = spirv.Constant 0 : i32
      %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %0 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %2 = spirv.Load "Input" %__builtin__WorkgroupId___addr_0 : vector<3xi32>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_1 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %4 = spirv.Load "Input" %__builtin__WorkgroupId___addr_1 : vector<3xi32>
      %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %5 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
      %__builtin__LocalInvocationId___addr_2 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %6 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_2 : vector<3xi32>
      %__builtin__LocalInvocationId___addr_3 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %7 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_3 : vector<3xi32>
      %__builtin__NumWorkgroups___addr = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %8 = spirv.Load "Input" %__builtin__NumWorkgroups___addr : vector<3xi32>
      %__builtin__NumWorkgroups___addr_4 = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %9 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_4 : vector<3xi32>
      %__builtin__NumWorkgroups___addr_5 = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %10 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_5 : vector<3xi32>
      %11 = spirv.IMul %1, %arg0 : i32
      %12 = spirv.IAdd %11, %arg1 : i32
      %13 = spirv.IMul %3, %arg0 : i32
      %14 = spirv.IAdd %13, %arg1 : i32
      %15 = spirv.IMul %12, %cst3_i32 : i32
      %16 = spirv.IAdd %14, %15 : i32
      %17 = spirv.AccessChain %arg2[%cst0_i32, %16] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %18 = spirv.Load "StorageBuffer" %17 : f32
      %19 = spirv.IMul %arg1, %cst3_i32 : i32
      %20 = spirv.IAdd %14, %19 : i32
      %21 = spirv.AccessChain %arg3[%cst0_i32, %20] : !spirv.ptr<!spirv.struct<(!spirv.array<3 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %22 = spirv.Load "StorageBuffer" %21 : f32
      %23 = spirv.FAdd %18, %22 : f32
      %24 = spirv.IMul %12, %cst3_i32 : i32
      %25 = spirv.IAdd %14, %24 : i32
      %26 = spirv.AccessChain %arg4[%cst0_i32, %25] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %26, %23 : f32
      spirv.Return
    }
  }
  gpu.module @arith.addf_linalg_kernel {
    gpu.func @arith.addf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg3: memref<1x3xf32, #spirv.storage_class<StorageBuffer>>, %arg4: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      %5 = memref.load %arg3[%arg1, %3] : memref<1x3xf32, #spirv.storage_class<StorageBuffer>>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      gpu.return
    }
  }
  func.func @arith.subf_linalg(%arg0: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) -> memref<2x3xf32, #spirv.storage_class<StorageBuffer>> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
    gpu.launch_func  @arith.subf_linalg_kernel::@arith.subf_linalg_kernel blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %arg0 : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>)
    return %alloc : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
  }
  spirv.module @__spv__arith.subf_linalg_kernel Logical GLSL450 {
    spirv.GlobalVariable @__builtin__NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.func @arith.subf_linalg_kernel(%arg0: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>}, %arg1: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1), StorageBuffer>}, %arg2: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}, %arg3: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 3)>}, %arg4: !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 4)>}) "None" attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>, workgroup_attributions = 0 : i64} {
      %cst3_i32 = spirv.Constant 3 : i32
      %cst0_i32 = spirv.Constant 0 : i32
      %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %0 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %2 = spirv.Load "Input" %__builtin__WorkgroupId___addr_0 : vector<3xi32>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_1 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %4 = spirv.Load "Input" %__builtin__WorkgroupId___addr_1 : vector<3xi32>
      %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %5 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
      %__builtin__LocalInvocationId___addr_2 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %6 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_2 : vector<3xi32>
      %__builtin__LocalInvocationId___addr_3 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %7 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_3 : vector<3xi32>
      %__builtin__NumWorkgroups___addr = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %8 = spirv.Load "Input" %__builtin__NumWorkgroups___addr : vector<3xi32>
      %__builtin__NumWorkgroups___addr_4 = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %9 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_4 : vector<3xi32>
      %__builtin__NumWorkgroups___addr_5 = spirv.mlir.addressof @__builtin__NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
      %10 = spirv.Load "Input" %__builtin__NumWorkgroups___addr_5 : vector<3xi32>
      %11 = spirv.IMul %1, %arg0 : i32
      %12 = spirv.IAdd %11, %arg1 : i32
      %13 = spirv.IMul %3, %arg0 : i32
      %14 = spirv.IAdd %13, %arg1 : i32
      %15 = spirv.IMul %12, %cst3_i32 : i32
      %16 = spirv.IAdd %14, %15 : i32
      %17 = spirv.AccessChain %arg2[%cst0_i32, %16] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %18 = spirv.Load "StorageBuffer" %17 : f32
      %19 = spirv.IMul %12, %cst3_i32 : i32
      %20 = spirv.IAdd %14, %19 : i32
      %21 = spirv.AccessChain %arg3[%cst0_i32, %20] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %22 = spirv.Load "StorageBuffer" %21 : f32
      %23 = spirv.FSub %18, %22 : f32
      %24 = spirv.IMul %12, %cst3_i32 : i32
      %25 = spirv.IAdd %14, %24 : i32
      %26 = spirv.AccessChain %arg4[%cst0_i32, %25] : !spirv.ptr<!spirv.struct<(!spirv.array<6 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %26, %23 : f32
      spirv.Return
    }
  }
  gpu.module @arith.subf_linalg_kernel {
    gpu.func @arith.subf_linalg_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg3: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>, %arg4: memref<2x3xf32, #spirv.storage_class<StorageBuffer>>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %arg0 overflow<nsw> : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 overflow<nsw> : index
      %3 = arith.addi %2, %arg1 : index
      %4 = memref.load %arg2[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      %5 = memref.load %arg3[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      %6 = arith.subf %4, %5 : f32
      memref.store %6, %arg4[%1, %3] : memref<2x3xf32, #spirv.storage_class<StorageBuffer>>
      gpu.return
    }
  }
}

