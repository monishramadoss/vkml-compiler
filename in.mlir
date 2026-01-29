#map = affine_map<()[s0, s1, s2] -> (s0 * s2 + s1)>
module attributes {gpu.container_module} {
  func.func @main() {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1)  args(%c1 : index, %c0 : index, %alloc : memref<2x3xf32>, %alloc_0 : memref<1x3xf32>, %alloc_1 : memref<2x3xf32>)
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32>, %arg3: memref<1x3xf32>, %arg4: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = affine.apply #map()[%arg0, %arg1, %block_id_x]
      %1 = affine.apply #map()[%arg0, %arg1, %block_id_y]
      %2 = memref.load %arg2[%0, %1] : memref<2x3xf32>
      %3 = memref.load %arg3[%arg1, %1] : memref<1x3xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg4[%0, %1] : memref<2x3xf32>
      gpu.return
    }
  }
}