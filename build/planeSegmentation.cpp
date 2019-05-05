#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <thread>


class planeSegmentationAndClustering
{
private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_blob; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented2; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z; 
  

public:
  planeSegmentationAndClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_blob, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented2){
    this->cloud_filtered = cloud_filtered;
    this->cloud_filtered_z = cloud_filtered_z;
    this->cloud_f = cloud_f;
    this->cloud_blob = cloud_blob;
    this->cloud_segmented = cloud_segmented;
    this->cloud_p = cloud_p;
    this->cloud_segmented2 = cloud_segmented2;



    
  }
    
  ~planeSegmentationAndClustering(){
  }
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  pcl::PCDWriter writer;
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

  pcl::PointCloud<pcl::PointXYZ>::Ptr getDownSampledCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_blob);
  void writeToDisk(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  void getSegmentedOutputs(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered, double distanceThreshold);
  void getClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented,double clusterThreshold);

};

pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::string s)
{

  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (s));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  // viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr planeSegmentationAndClustering::getDownSampledCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_blob){ 
  
  sor.setInputCloud (cloud_blob);
  
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered);
  return cloud_filtered;
}



void planeSegmentationAndClustering::writeToDisk(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  writer.write<pcl::PointXYZ> ("outputs/downsampled_scene_cloud.pcd", *cloud, false);

}

void planeSegmentationAndClustering::getSegmentedOutputs(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered, double distanceThreshold){
  

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory

  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_MSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (distanceThreshold);

  
  int count(5);

  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  // While 30% of the original cloud is still there
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*cloud_p);
    std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;
    std::stringstream ss;

    
    ss << "outputs/Plane_Segmented_Output_" << i << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_p, false);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
    count--;
  }
}

void planeSegmentationAndClustering::getClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented, double clusterThreshold){
  
  
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_segmented);
  *cloud_filtered = *cloud_segmented;

  std::vector<pcl::PointIndices> cluster_indices;
  
  ec.setClusterTolerance (clusterThreshold); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "outputs/cloud_cluster_" << j << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
    j++;
  }
}



int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_blob (new pcl::PointCloud<pcl::PointXYZ>); 
 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented2 (new pcl::PointCloud<pcl::PointXYZ>);

  planeSegmentationAndClustering pc(cloud_filtered, cloud_blob, cloud_p, cloud_segmented, cloud_filtered_z, cloud_f, cloud_segmented2);

  // Fill in the cloud data
  pcl::PCDReader reader;
  pcl::PCDWriter writer;
  reader.read (argv[1], *cloud_blob);

  std::cerr << "PointCloud before filtering: " << cloud_blob->width * cloud_blob->height << " data points." << std::endl;
  
  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  cloud_filtered = pc.getDownSampledCloud(cloud_blob);
  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

  // Write the downsampled version to disk
  pc.writeToDisk(cloud_filtered);

  //Write the segmented table outputs to disk
  double distanceThreshold = atof(argv[2]);
  pc.getSegmentedOutputs(cloud_filtered, distanceThreshold);
  //Save the output after the
  reader.read("outputs/Plane_Segmented_Output_0.pcd", *cloud_segmented);



  //Get and Write Clusters of plane segmented object
  pc.getClusters(cloud_segmented, 0.02);




  pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr1 (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr2 (new pcl::PointCloud<pcl::PointXYZ>);

  reader.read("outputs/cloud_cluster_0.pcd", *basic_cloud_ptr1);
  reader.read("outputs/cloud_cluster_1.pcd", *basic_cloud_ptr2);

  pcl::visualization::PCLVisualizer::Ptr viewer5;
  pcl::visualization::PCLVisualizer::Ptr viewer6;

  viewer5 = simpleVis(basic_cloud_ptr2,"cloud_cluster_1");
  viewer6 = simpleVis(basic_cloud_ptr1,"cloud_cluster_0");
  int condition(0);
  int clusterNum(0); //default cluster to clusterized is 0

  while (!viewer6->wasStopped ())
  {

    viewer5->spinOnce (100);
    viewer6->spinOnce (100);
    std::cout << "Do you want to recluster the zeroth cluster? Enter 1 if you want to." << std::endl;
    std::cin >> condition;
    


    if(condition == 0 || 1)
    	break;

   }


  if(condition)
  {
  	  //view the first two clusters
  	

  	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr3 (new pcl::PointCloud<pcl::PointXYZ>);
  	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr4 (new pcl::PointCloud<pcl::PointXYZ>);
  	std::stringstream ss;

  	std::cout << "Enter the Cluster Number that you wish to recluster i.e. either 0 or 1" << std::endl;
  	std::cin >> clusterNum;

  	ss << "outputs/cloud_cluster_" << clusterNum << ".pcd";
  	std::cout << ss.str() << std::endl;



  	reader.read(ss.str(), *cloud_segmented2);
  	double clusterThreshold = atof(argv[3]);
  	pc.getClusters(cloud_segmented2, clusterThreshold);


  	reader.read("outputs/cloud_cluster_0.pcd", *basic_cloud_ptr1);
  	reader.read("outputs/cloud_cluster_1.pcd", *basic_cloud_ptr2);
  	reader.read("outputs/cloud_cluster_2.pcd", *basic_cloud_ptr3);
  	reader.read("outputs/Plane_Segmented_Output_0.pcd", *basic_cloud_ptr4);


  	pcl::visualization::PCLVisualizer::Ptr viewer1;
  	pcl::visualization::PCLVisualizer::Ptr viewer2;
  	pcl::visualization::PCLVisualizer::Ptr viewer3;
  	pcl::visualization::PCLVisualizer::Ptr viewer4;



  	viewer1 = simpleVis(basic_cloud_ptr4, "Plane_Segmented_Output_0");
  	viewer2 = simpleVis(basic_cloud_ptr3, "cloud_cluster_2");
  	viewer3 = simpleVis(basic_cloud_ptr2,"cloud_cluster_1");
  	viewer4 = simpleVis(basic_cloud_ptr1,"cloud_cluster_0");


  	while (!viewer4->wasStopped ())
  	{
  		viewer4->spinOnce (100);
  		viewer3->spinOnce (100);
  		viewer2->spinOnce (100);
  		viewer1->spinOnce (100);


    //std::this_thread::sleep_for(100);
  	}

  }





  







  return (0);
}


  