import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from collections import defaultdict
import glob
import json
from numbers import Number

def analyze_single_dataset(dataset_path):
    mhd_files = glob.glob(os.path.join(dataset_path, "**/*.mhd"), recursive=True)
    
    if not mhd_files:
        return None
    
    stats = {
        'total_scans': len(mhd_files),
        'dimensions': [],
        'spacings': [],
        'slice_thicknesses': [],
        'pixel_types': [],
        'voxel_counts': [],
        'physical_sizes': [],
        'orientations': [],
        'intensity_stats': defaultdict(list),
        'file_sizes': []
    }
    
    successful_analyses = 0
    for i, mhd_file in enumerate(mhd_files):
        try:
            image = sitk.ReadImage(mhd_file)
            
            dimensions = image.GetSize()
            stats['dimensions'].append(dimensions)
            
            spacing = image.GetSpacing()
            stats['spacings'].append(spacing)
            stats['slice_thicknesses'].append(spacing[2])
            
            pixel_type = image.GetPixelIDTypeAsString()
            stats['pixel_types'].append(pixel_type)
            
            total_voxels = dimensions[0] * dimensions[1] * dimensions[2]
            stats['voxel_counts'].append(total_voxels)
            
            physical_size = (
                dimensions[0] * spacing[0],
                dimensions[1] * spacing[1], 
                dimensions[2] * spacing[2]
            )
            stats['physical_sizes'].append(physical_size)
            
            direction = image.GetDirection()
            stats['orientations'].append(direction)
            
            mhd_size = os.path.getsize(mhd_file)
            raw_file = mhd_file.replace('.mhd', '.zraw')
            if os.path.exists(raw_file):
                raw_size = os.path.getsize(raw_file)
                stats['file_sizes'].append(mhd_size + raw_size)
            else:
                stats['file_sizes'].append(mhd_size)
            
            array = sitk.GetArrayFromImage(image)
            stats['intensity_stats']['min'].append(np.min(array))
            stats['intensity_stats']['max'].append(np.max(array))
            stats['intensity_stats']['mean'].append(np.mean(array))
            stats['intensity_stats']['std'].append(np.std(array))
            stats['intensity_stats']['median'].append(np.median(array))
            
            successful_analyses += 1
            
        except Exception as e:
            continue
    
    return stats

def compute_global_characteristics(individual_stats):
    if not individual_stats or not individual_stats['dimensions']:
        return None
    
    global_stats = {
        'dataset_name': '',
        'dataset_path': '',
        'dataset_summary': {},
        'dimensions_global': {},
        'spacing_global': {},
        'physical_size_global': {},
        'intensity_global': {},
        'data_characteristics': {},
        'ranges': {}
    }
    
    dimensions_array = np.array(individual_stats['dimensions'])
    spacing_array = np.array(individual_stats['spacings'])
    voxel_counts = np.array(individual_stats['voxel_counts'])
    file_sizes_mb = np.array(individual_stats['file_sizes']) / (1024 * 1024)
    
    physical_sizes = []
    for i in range(len(individual_stats['dimensions'])):
        dims = individual_stats['dimensions'][i]
        spac = individual_stats['spacings'][i]
        physical_sizes.append([dims[0] * spac[0], dims[1] * spac[1], dims[2] * spac[2]])
    physical_sizes_array = np.array(physical_sizes)
    
    global_stats['dataset_summary'] = {
        'total_scans': individual_stats['total_scans'],
        'successful_analyses': len(individual_stats['dimensions']),
        'total_voxels': np.sum(voxel_counts),
        'avg_file_size_mb': np.mean(file_sizes_mb),
        'total_disk_size_gb': np.sum(file_sizes_mb) / 1024
    }
    
    global_stats['dimensions_global'] = {
        'width_mean': float(np.mean(dimensions_array[:, 0])),
        'width_std': float(np.std(dimensions_array[:, 0])),
        'width_min': int(np.min(dimensions_array[:, 0])),
        'width_max': int(np.max(dimensions_array[:, 0])),
        'height_mean': float(np.mean(dimensions_array[:, 1])),
        'height_std': float(np.std(dimensions_array[:, 1])),
        'height_min': int(np.min(dimensions_array[:, 1])),
        'height_max': int(np.max(dimensions_array[:, 1])),
        'slices_mean': float(np.mean(dimensions_array[:, 2])),
        'slices_std': float(np.std(dimensions_array[:, 2])),
        'slices_min': int(np.min(dimensions_array[:, 2])),
        'slices_max': int(np.max(dimensions_array[:, 2])),
        'typical_dimensions': f"{int(np.mean(dimensions_array[:, 0]))}x{int(np.mean(dimensions_array[:, 1]))}x{int(np.mean(dimensions_array[:, 2]))}"
    }
    
    global_stats['spacing_global'] = {
        'x_resolution_mean': float(np.mean(spacing_array[:, 0])),
        'x_resolution_std': float(np.std(spacing_array[:, 0])),
        'x_resolution_min': float(np.min(spacing_array[:, 0])),
        'x_resolution_max': float(np.max(spacing_array[:, 0])),
        'y_resolution_mean': float(np.mean(spacing_array[:, 1])),
        'y_resolution_std': float(np.std(spacing_array[:, 1])),
        'y_resolution_min': float(np.min(spacing_array[:, 1])),
        'y_resolution_max': float(np.max(spacing_array[:, 1])),
        'slice_thickness_mean': float(np.mean(spacing_array[:, 2])),
        'slice_thickness_std': float(np.std(spacing_array[:, 2])),
        'slice_thickness_min': float(np.min(spacing_array[:, 2])),
        'slice_thickness_max': float(np.max(spacing_array[:, 2])),
        'typical_spacing': f"{np.mean(spacing_array[:, 0]):.3f}x{np.mean(spacing_array[:, 1]):.3f}x{np.mean(spacing_array[:, 2]):.3f} mm"
    }
    
    global_stats['physical_size_global'] = {
        'width_mm_mean': float(np.mean(physical_sizes_array[:, 0])),
        'width_mm_std': float(np.std(physical_sizes_array[:, 0])),
        'width_mm_min': float(np.min(physical_sizes_array[:, 0])),
        'width_mm_max': float(np.max(physical_sizes_array[:, 0])),
        'height_mm_mean': float(np.mean(physical_sizes_array[:, 1])),
        'height_mm_std': float(np.std(physical_sizes_array[:, 1])),
        'height_mm_min': float(np.min(physical_sizes_array[:, 1])),
        'height_mm_max': float(np.max(physical_sizes_array[:, 1])),
        'depth_mm_mean': float(np.mean(physical_sizes_array[:, 2])),
        'depth_mm_std': float(np.std(physical_sizes_array[:, 2])),
        'depth_mm_min': float(np.min(physical_sizes_array[:, 2])),
        'depth_mm_max': float(np.max(physical_sizes_array[:, 2]))
    }
    
    intensity_min = np.array(individual_stats['intensity_stats']['min'])
    intensity_max = np.array(individual_stats['intensity_stats']['max'])
    intensity_mean = np.array(individual_stats['intensity_stats']['mean'])
    intensity_std = np.array(individual_stats['intensity_stats']['std'])
    intensity_median = np.array(individual_stats['intensity_stats']['median'])
    
    global_stats['intensity_global'] = {
        'min_mean': float(np.mean(intensity_min)),
        'min_std': float(np.std(intensity_min)),
        'min_range_min': float(np.min(intensity_min)),
        'min_range_max': float(np.max(intensity_min)),
        'max_mean': float(np.mean(intensity_max)),
        'max_std': float(np.std(intensity_max)),
        'max_range_min': float(np.min(intensity_max)),
        'max_range_max': float(np.max(intensity_max)),
        'mean_intensity_mean': float(np.mean(intensity_mean)),
        'mean_intensity_std': float(np.std(intensity_mean)),
        'std_mean': float(np.mean(intensity_std)),
        'median_mean': float(np.mean(intensity_median)),
        'typical_range': f"{int(np.mean(intensity_min))} - {int(np.mean(intensity_max))} HU"
    }
    
    pixel_types = individual_stats['pixel_types']
    pixel_type_counts = {pt: pixel_types.count(pt) for pt in set(pixel_types)}
    
    global_stats['data_characteristics'] = {
        'main_pixel_type': max(pixel_type_counts, key=pixel_type_counts.get),
        'pixel_type_variants': len(pixel_type_counts),
        'voxels_per_scan_mean': float(np.mean(voxel_counts)),
        'voxels_per_scan_std': float(np.std(voxel_counts)),
        'voxels_per_scan_min': int(np.min(voxel_counts)),
        'voxels_per_scan_max': int(np.max(voxel_counts)),
        'consistent_dimensions': len(set([tuple(dim) for dim in individual_stats['dimensions']])) == 1,
        'consistent_spacing': len(set([tuple(spc) for spc in individual_stats['spacings']])) == 1,
        'consistent_pixel_type': len(set(pixel_types)) == 1
    }
    
    global_stats['ranges'] = {
        'slices_range': f"{int(np.min(dimensions_array[:, 2]))} - {int(np.max(dimensions_array[:, 2]))}",
        'resolution_range_xy': f"{np.min(spacing_array[:, 0]):.3f}-{np.max(spacing_array[:, 0]):.3f} mm",
        'resolution_range_z': f"{np.min(spacing_array[:, 2]):.3f}-{np.max(spacing_array[:, 2]):.3f} mm",
        'physical_size_range_x': f"{np.min(physical_sizes_array[:, 0]):.1f}-{np.max(physical_sizes_array[:, 0]):.1f} mm",
        'physical_size_range_y': f"{np.min(physical_sizes_array[:, 1]):.1f}-{np.max(physical_sizes_array[:, 1]):.1f} mm",
        'physical_size_range_z': f"{np.min(physical_sizes_array[:, 2]):.1f}-{np.max(physical_sizes_array[:, 2]):.1f} mm"
    }
    
    return global_stats

def save_global_characteristics(global_stats, dataset_name, dataset_path):
    if not global_stats:
        return

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets_utils"))
    os.makedirs(save_dir, exist_ok=True)

    global_stats['dataset_name'] = dataset_name
    global_stats['dataset_path'] = dataset_path

    def make_serializable(o):
        if isinstance(o, dict):
            return {k: make_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [make_serializable(x) for x in o]
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, Number):
            return o
        return str(o)

    json_path = os.path.join(save_dir, f"{dataset_name}_global_characteristics.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(make_serializable(global_stats), jf, indent=2, ensure_ascii=False)

    csv_data = {
        'dataset_name': dataset_name,
        'dataset_path': dataset_path,
        'total_scans': global_stats['dataset_summary']['total_scans'],
        'successful_analyses': global_stats['dataset_summary']['successful_analyses'],
        'total_voxels': global_stats['dataset_summary']['total_voxels'],
        'avg_file_size_mb': global_stats['dataset_summary']['avg_file_size_mb'],
        'total_disk_size_gb': global_stats['dataset_summary']['total_disk_size_gb'],
        'width_mean': global_stats['dimensions_global']['width_mean'],
        'width_std': global_stats['dimensions_global']['width_std'],
        'width_min': global_stats['dimensions_global']['width_min'],
        'width_max': global_stats['dimensions_global']['width_max'],
        'height_mean': global_stats['dimensions_global']['height_mean'],
        'height_std': global_stats['dimensions_global']['height_std'],
        'height_min': global_stats['dimensions_global']['height_min'],
        'height_max': global_stats['dimensions_global']['height_max'],
        'slices_mean': global_stats['dimensions_global']['slices_mean'],
        'slices_std': global_stats['dimensions_global']['slices_std'],
        'slices_min': global_stats['dimensions_global']['slices_min'],
        'slices_max': global_stats['dimensions_global']['slices_max'],
        'typical_dimensions': global_stats['dimensions_global']['typical_dimensions'],
        'x_resolution_mean': global_stats['spacing_global']['x_resolution_mean'],
        'x_resolution_std': global_stats['spacing_global']['x_resolution_std'],
        'x_resolution_min': global_stats['spacing_global']['x_resolution_min'],
        'x_resolution_max': global_stats['spacing_global']['x_resolution_max'],
        'y_resolution_mean': global_stats['spacing_global']['y_resolution_mean'],
        'y_resolution_std': global_stats['spacing_global']['y_resolution_std'],
        'y_resolution_min': global_stats['spacing_global']['y_resolution_min'],
        'y_resolution_max': global_stats['spacing_global']['y_resolution_max'],
        'slice_thickness_mean': global_stats['spacing_global']['slice_thickness_mean'],
        'slice_thickness_std': global_stats['spacing_global']['slice_thickness_std'],
        'slice_thickness_min': global_stats['spacing_global']['slice_thickness_min'],
        'slice_thickness_max': global_stats['spacing_global']['slice_thickness_max'],
        'typical_spacing': global_stats['spacing_global']['typical_spacing'],
        'width_mm_mean': global_stats['physical_size_global']['width_mm_mean'],
        'width_mm_std': global_stats['physical_size_global']['width_mm_std'],
        'width_mm_min': global_stats['physical_size_global']['width_mm_min'],
        'width_mm_max': global_stats['physical_size_global']['width_mm_max'],
        'height_mm_mean': global_stats['physical_size_global']['height_mm_mean'],
        'height_mm_std': global_stats['physical_size_global']['height_mm_std'],
        'height_mm_min': global_stats['physical_size_global']['height_mm_min'],
        'height_mm_max': global_stats['physical_size_global']['height_mm_max'],
        'depth_mm_mean': global_stats['physical_size_global']['depth_mm_mean'],
        'depth_mm_std': global_stats['physical_size_global']['depth_mm_std'],
        'depth_mm_min': global_stats['physical_size_global']['depth_mm_min'],
        'depth_mm_max': global_stats['physical_size_global']['depth_mm_max'],
        'intensity_min_mean': global_stats['intensity_global']['min_mean'],
        'intensity_min_std': global_stats['intensity_global']['min_std'],
        'intensity_max_mean': global_stats['intensity_global']['max_mean'],
        'intensity_max_std': global_stats['intensity_global']['max_std'],
        'intensity_mean_mean': global_stats['intensity_global']['mean_intensity_mean'],
        'intensity_mean_std': global_stats['intensity_global']['mean_intensity_std'],
        'intensity_std_mean': global_stats['intensity_global']['std_mean'],
        'intensity_median_mean': global_stats['intensity_global']['median_mean'],
        'typical_intensity_range': global_stats['intensity_global']['typical_range'],
        'main_pixel_type': global_stats['data_characteristics']['main_pixel_type'],
        'pixel_type_variants': global_stats['data_characteristics']['pixel_type_variants'],
        'voxels_per_scan_mean': global_stats['data_characteristics']['voxels_per_scan_mean'],
        'voxels_per_scan_std': global_stats['data_characteristics']['voxels_per_scan_std'],
        'consistent_dimensions': global_stats['data_characteristics']['consistent_dimensions'],
        'consistent_spacing': global_stats['data_characteristics']['consistent_spacing'],
        'consistent_pixel_type': global_stats['data_characteristics']['consistent_pixel_type']
    }

    df = pd.DataFrame([csv_data])
    csv_path = os.path.join(save_dir, f"{dataset_name}_global_characteristics.csv")
    df.to_csv(csv_path, index=False)

def analyze_specific_dataset():
    dataset_path = "fra_folder"
    dataset_name = "fra"
    
    if not os.path.exists(dataset_path):
        return
    
    stats = analyze_single_dataset(dataset_path)
    
    if stats:
        global_stats = compute_global_characteristics(stats)
        
        if global_stats:
            save_global_characteristics(global_stats, dataset_name, dataset_path)

analyze_specific_dataset()