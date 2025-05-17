# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
from physics_core.system import ChargeSystem
from physics_core.distributions import DISTRIBUTION_CLASSES # Useful for validation messages

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True # For cleaner JSON output

# Helper to validate range and resolution parameters
def _parse_range_res(param_name, data, default_min=-5, default_max=5, default_res=10):
    """Parses range (min, max) and resolution for one axis."""
    p_range = data.get(f"{param_name}_range")
    p_res = data.get(f"{param_name}_resolution")

    if p_range is None: p_range = [default_min, default_max]
    if p_res is None: p_res = default_res

    if not (isinstance(p_range, list) and len(p_range) == 2 and all(isinstance(x, (int, float)) for x in p_range)):
        raise ValueError(f"Invalid {param_name}_range format. Expected list like [-5, 5].")
    if not (isinstance(p_res, int) and p_res >= 2): # Min resolution 2 points for a range
        raise ValueError(f"Invalid {param_name}_resolution. Expected integer >= 2.")

    # Handle case where min >= max
    if p_range[0] >= p_range[1]:
         # Option 1: Swap them. Option 2: Return a single point array.
         # Let's return a single point array at the min value.
         # A resolution of 1 is essentially a point, but we enforce >=2 for linspace.
         # If resolution is 2, but range is [0,0], linspace(0,0,2) gives [0,0]. This is fine.
         if p_range[0] == p_range[1]:
             return np.array([p_range[0]]) # Single point array

         # If min > max, swap
         print(f"Warning: {param_name}_range min > max. Swapping values.")
         p_range = [p_range[1], p_range[0]]

    # Use np.linspace to generate coordinates
    coords = np.linspace(p_range[0], p_range[1], p_res)
    # Add a small tolerance if point is exactly on a singularity (e.g. point charge location)
    # This is tricky and might be better handled within the get_electric_field methods
    # For now, relying on the 1e-9 or 1e-12 checks in distributions.py
    # coords += np.random.rand(*coords.shape) * 1e-9 # small random jitter - not ideal

    return coords


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/calculate_field_at_point', methods=['POST'])
def calculate_field_at_point():
    """API endpoint to calculate E-field and potential at a single point."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        distributions_data = data.get('distributions')
        point_coords = data.get('point') # Expected: {"x": val, "y": val, "z": val}

        if not distributions_data or not isinstance(distributions_data, list):
            return jsonify({"error": "Missing or invalid 'distributions' list"}), 400
        if not point_coords or not all(k in point_coords for k in ['x', 'y', 'z']):
            return jsonify({"error": "Missing or invalid 'point' coordinates (expected {'x': v, 'y': v, 'z': v})"}), 400

        # Basic check if point_coords values are numbers
        if not all(isinstance(point_coords[k], (int, float)) for k in ['x', 'y', 'z']):
             return jsonify({"error": "'point' coordinates must be numbers"}), 400


        system = ChargeSystem.from_dict_list(distributions_data)

        x, y, z = point_coords['x'], point_coords['y'], point_coords['z']
        E_field = system.calculate_total_electric_field(x, y, z)
        potential = system.calculate_total_potential(x,y,z)

        # Use float() to ensure numpy floats are serialized correctly to JSON
        return jsonify({
            "Ex": float(E_field[0]), "Ey": float(E_field[1]), "Ez": float(E_field[2]),
            "E_magnitude": float(np.linalg.norm(E_field)),
            "potential": float(potential)
        })

    except ValueError as ve: # Catch errors from distribution creation or parsing
        app.logger.error(f"ValueError in /api/calculate_field_at_point: {ve}")
        available_types_str = ', '.join(DISTRIBUTION_CLASSES.keys())
        error_msg = str(ve)
        if "Unknown or missing distribution type" in error_msg:
             error_msg += f". Available types: {available_types_str}"
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/calculate_field_at_point: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred"}), 500


@app.route('/api/calculate_field_on_slice', methods=['POST'])
def calculate_field_on_slice():
    """API endpoint to calculate E-field on a 2D slice (grid)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        distributions_data = data.get('distributions')
        slice_def = data.get('slice_definition') # e.g., {"axis": "z", "value": 0, "x_range": [-5,5], "y_range": [-5,5], "resolution": 20}

        if not distributions_data or not isinstance(distributions_data, list):
            return jsonify({"error": "Missing or invalid 'distributions' list"}), 400
        if not slice_def:
            return jsonify({"error": "Missing 'slice_definition'"}), 400

        system = ChargeSystem.from_dict_list(distributions_data)

        axis = slice_def.get('axis', 'z').lower()
        value = slice_def.get('value', 0.0)
        resolution = slice_def.get('resolution', 20) # Default resolution for both dimensions

        # Set ranges for the two perpendicular axes based on the slice axis
        if axis == 'x':
            x_coords = np.array([value]) # Fixed coordinate on the slice axis
            y_coords = _parse_range_res('y', slice_def, default_res=resolution)
            z_coords = _parse_range_res('z', slice_def, default_res=resolution)
            primary_axis_idx, secondary_axis_idx = 1, 2 # y, z
        elif axis == 'y':
            y_coords = np.array([value]) # Fixed coordinate on the slice axis
            x_coords = _parse_range_res('x', slice_def, default_res=resolution)
            z_coords = _parse_range_res('z', slice_def, default_res=resolution)
            primary_axis_idx, secondary_axis_idx = 0, 2 # x, z
        elif axis == 'z':
            z_coords = np.array([value]) # Fixed coordinate on the slice axis
            x_coords = _parse_range_res('x', slice_def, default_res=resolution)
            y_coords = _parse_range_res('y', slice_def, default_res=resolution)
            primary_axis_idx, secondary_axis_idx = 0, 1 # x, y
        else:
            return jsonify({"error": "Invalid slice axis. Must be 'x', 'y', or 'z'."}), 400

        # Calculate fields on the grid points
        points, vectors = system.calculate_field_on_grid(x_coords, y_coords, z_coords)

        # For 2D slice, we typically want the 2D coordinates and 2D components of the vector
        # that lie in that plane for plotting.
        plot_points_dim1 = points[:, primary_axis_idx].tolist()
        plot_points_dim2 = points[:, secondary_axis_idx].tolist()
        plot_vectors_dim1 = vectors[:, primary_axis_idx].tolist()
        plot_vectors_dim2 = vectors[:, secondary_axis_idx].tolist()

        return jsonify({
            "points_dim1": plot_points_dim1,
            "points_dim2": plot_points_dim2,
            "vectors_dim1": plot_vectors_dim1,
            "vectors_dim2": plot_vectors_dim2,
            # Provide labels corresponding to the plotted dimensions
            "axis_labels": [chr(ord('x') + primary_axis_idx), chr(ord('x') + secondary_axis_idx)],
            "slice_axis": axis,
            "slice_value": float(value)
        })

    except ValueError as ve:
        app.logger.error(f"ValueError in /api/calculate_field_on_slice: {ve}")
        available_types_str = ', '.join(DISTRIBUTION_CLASSES.keys())
        error_msg = str(ve)
        if "Unknown or missing distribution type" in error_msg:
             error_msg += f". Available types: {available_types_str}"
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/calculate_field_on_slice: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred"}), 500


@app.route('/api/calculate_field_in_volume', methods=['POST'])
def calculate_field_in_volume():
    """API endpoint to calculate E-field on a 3D volume (grid)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        distributions_data = data.get('distributions')
        volume_def = data.get('volume_definition') # e.g., {"x_range": [-5,5], "y_range": [-5,5], "z_range": [-5,5], "resolution": 10}

        if not distributions_data or not isinstance(distributions_data, list):
            return jsonify({"error": "Missing or invalid 'distributions' list"}), 400
        if not volume_def:
            return jsonify({"error": "Missing 'volume_definition'"}), 400

        system = ChargeSystem.from_dict_list(distributions_data)

        # Allow a single 'resolution' value for all axes, or per-axis resolution
        # If 'resolution' is present, use it for axes not explicitly specified
        default_res = volume_def.get("resolution", 10)

        x_coords = _parse_range_res('x', volume_def, default_res=default_res)
        y_coords = _parse_range_res('y', volume_def, default_res=default_res)
        z_coords = _parse_range_res('z', volume_def, default_res=default_res)

        points, vectors = system.calculate_field_on_grid(x_coords, y_coords, z_coords)

        # Prepare data for Plotly cone plot
        return jsonify({
            "points_x": points[:, 0].tolist(),
            "points_y": points[:, 1].tolist(),
            "points_z": points[:, 2].tolist(),
            "vectors_u": vectors[:, 0].tolist(), # Corresponds to x-component of E
            "vectors_v": vectors[:, 1].tolist(), # Corresponds to y-component of E
            "vectors_w": vectors[:, 2].tolist()  # Corresponds to z-component of E
        })

    except ValueError as ve:
        app.logger.error(f"ValueError in /api/calculate_field_in_volume: {ve}")
        available_types_str = ', '.join(DISTRIBUTION_CLASSES.keys())
        error_msg = str(ve)
        if "Unknown or missing distribution type" in error_msg:
             error_msg += f". Available types: {available_types_str}"
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/calculate_field_in_volume: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    # In production, use a production-ready WSGI server like Gunicorn or uWSGI
    # For development, debug=True is useful
    app.run(debug=True)