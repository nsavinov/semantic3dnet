add_library(pointCloudUtil SHARED point_cloud_util.cpp data_loader_wrapper.cpp box3D.cpp voxel_grid3D.cpp) #SHARED
set_target_properties(pointCloudUtil PROPERTIES LINK_FLAGS "-fopenmp")
