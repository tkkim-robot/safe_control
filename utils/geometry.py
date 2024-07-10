from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import explain_validity

def custom_merge(geometries):
    # First, perform a union operation
    union = unary_union(geometries)
    
    # If the result is a MultiPolygon, we need to process each polygon
    if isinstance(union, MultiPolygon):
        processed_polygons = [process_polygon(poly, geometries) for poly in union.geoms]
        merged_polygon =  MultiPolygon(processed_polygons)
    elif isinstance(union, Polygon):
        merged_polygon = process_polygon(union, geometries)
    else:
        merged_polygon = union  # In case it's neither Polygon nor MultiPolygon
    
    # Fix geometry if necessary
    merged_polygon = fix_geometry(merged_polygon)
    return merged_polygon

def process_polygon(polygon, original_geometries):
    exterior = polygon.exterior
    valid_interiors = []
    
    for interior in polygon.interiors:
        if should_keep_interior(interior, original_geometries, polygon):
            valid_interiors.append(interior)
    
    return Polygon(exterior, valid_interiors)

def should_keep_interior(interior, original_geometries, merged_area):
    interior_poly = Polygon(interior)
    
    # Check if this interior is part of any original geometry's exterior
    for geom in original_geometries:
        if isinstance(geom, Polygon):
            if geom.exterior.contains(interior_poly):
                return False
        elif isinstance(geom, MultiPolygon):
            if any(poly.exterior.contains(interior_poly) for poly in geom.geoms):
                return False
    
    # # Check if this interior is properly contained within the merged area
    # if not merged_area.contains(interior_poly):
    #     return False
    
    return True
    
def fix_geometry(geometry, buffer_distance=0.2):
    if not geometry.is_valid:
        reason = explain_validity(geometry)
        print(reason)
        if "Self-intersection" in reason:
            # Apply a small positive buffer followed by a small negative buffer
            geometry = geometry.buffer(buffer_distance).buffer(-buffer_distance)
        if "Holes are nested" in reason:
            print(f"Fixing nested holes: {reason}")
            # Apply a small positive buffer to the exterior
            geometry = fix_nested_holes(geometry)
        if "Hole lies outside shell" in reason:
            print(f"Fixing holes outside shell: {reason}")
            # Remove holes that are outside the shell
            geometry = fix_holes(geometry)
        
    # if geometry is still invalid, try a more aggresive fix
    # if not geometry.is_valid:
    #     geometry = geometry.buffer(0)
    return geometry

def fix_nested_holes(geometry):
    '''
    Fix "Holes are nested" error by removing nested holes (recursive function)
    '''
    if isinstance(geometry, Polygon):
        exterior = geometry.exterior
        interiors = list(geometry.interiors)
        non_nested_interiors = []

        while interiors:
            interior = interiors.pop(0)
            if all(not inner.contains(interior) for inner in interiors):
                non_nested_interiors.append(interior)

        return Polygon(exterior, non_nested_interiors)
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([fix_nested_holes(geom) for geom in geometry.geoms])
    else:
        return geometry
    
def fix_holes(geometry):
    '''
    Fix "Hole lies outside shell" error by removing holes that are outside the shell (recursive function)
    '''
    if isinstance(geometry, Polygon):
        exterior = geometry.exterior
        valid_interiors = []
        for interior in geometry.interiors:
            if Polygon(exterior).contains(Polygon(interior)):
                valid_interiors.append(interior)
        return Polygon(exterior, valid_interiors)
    elif isinstance(geometry, MultiPolygon):
        fixed_polygons = [fix_holes(geom) for geom in geometry.geoms]
        return MultiPolygon(fixed_polygons)
    else:
        return geometry